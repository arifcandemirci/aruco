#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>

#include <libcamera/libcamera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/framebuffer_allocator.h>
#include <libcamera/stream.h>
#include <libcamera/formats.h>
#include <libcamera/control_ids.h>

#include <sys/mman.h>
#include <unistd.h>

#include <iostream>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <atomic>
#include <memory>

using namespace libcamera;

struct FpsCounter {
    double fps = 0.0;
    int cnt = 0;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

    void draw(cv::Mat &img) {
        cnt++;
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - t0).count();
        if (elapsed >= 0.5) {
            fps = cnt / elapsed;
            cnt = 0;
            t0 = now;
        }
        cv::putText(img, "FPS:" + cv::format("%.1f", fps), cv::Point(10, 40),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
    }
};

static cv::Mat yuv420_to_bgr(const uint8_t *yuv, int w, int h) {
    // I420 (YUV420p) -> BGR
    cv::Mat yuvImg(h + h / 2, w, CV_8UC1, const_cast<uint8_t*>(yuv));
    cv::Mat bgr;
    cv::cvtColor(yuvImg, bgr, cv::COLOR_YUV2BGR_I420);
    return bgr;
}

static cv::Mat y_plane_as_gray_from_yuv420(const uint8_t *yuv, int w, int h) {
    // Y plane first w*h bytes
    cv::Mat gray(h, w, CV_8UC1, const_cast<uint8_t*>(yuv));
    return gray.clone(); // copy out (buffer reuse olacağı için)
}

struct DualStreamState {
    // main (BGR) ve lores (gray) frame’leri senkron tut
    cv::Mat main_bgr;
    cv::Mat lores_gray;
    bool got_main = false;
    bool got_lores = false;

    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<bool> running{true};
};

static void *mmap_plane(const FrameBuffer::Plane &plane) {
    int fd = plane.fd.get();
    size_t length = plane.length;
    void *mem = mmap(nullptr, length, PROT_READ, MAP_SHARED, fd, 0);
    if (mem == MAP_FAILED) return nullptr;
    return mem;
}

static void munmap_plane(void *mem, const FrameBuffer::Plane &plane) {
    if (!mem) return;
    munmap(mem, plane.length);
}

int main() {
    // -------- libcamera setup --------
    std::unique_ptr<CameraManager> cm = std::make_unique<CameraManager>();
    if (cm->start()) {
        std::cerr << "CameraManager start failed\n";
        return 1;
    }
    if (cm->cameras().empty()) {
        std::cerr << "No camera found\n";
        return 1;
    }

    std::shared_ptr<Camera> cam = cm->cameras()[0];
    if (cam->acquire()) {
        std::cerr << "Camera acquire failed\n";
        return 1;
    }

    // 2 stream: main + lores (Viewfinder + StillCapture yerine; role’lar değişebilir)
    std::unique_ptr<CameraConfiguration> config =
        cam->generateConfiguration({StreamRole::Viewfinder, StreamRole::Viewfinder});

    if (!config || config->size() < 2) {
        std::cerr << "Failed to generate dual stream config\n";
        return 1;
    }

    // main: 320x240 RGB/BGR için (biz YUV420 alıp BGR’e çeviriyoruz; pratik)
    StreamConfiguration &mainCfg = config->at(0);
    mainCfg.size.width = 320;
    mainCfg.size.height = 240;
    mainCfg.pixelFormat = formats::YUV420; // daha uyumlu; BGR’e çevireceğiz

    // lores: 320x240 YUV420 (analiz)
    StreamConfiguration &loresCfg = config->at(1);
    loresCfg.size.width = 320;
    loresCfg.size.height = 240;
    loresCfg.pixelFormat = formats::YUV420;

    if (config->validate() == CameraConfiguration::Invalid) {
        std::cerr << "Config invalid\n";
        return 1;
    }
    if (cam->configure(config.get())) {
        std::cerr << "Camera configure failed\n";
        return 1;
    }

    Stream *mainStream = mainCfg.stream();
    Stream *loresStream = loresCfg.stream();

    FrameBufferAllocator allocator(cam);
    if (allocator.allocate(mainStream) < 0 || allocator.allocate(loresStream) < 0) {
        std::cerr << "Allocator failed\n";
        return 1;
    }

    // 30 FPS hedefi için frame duration (microseconds)
    // 30 fps => 33333 us. Python’daki 8888 us aslında ~112 fps.
    // Yine de “sabit fps” istiyorsan 33333 yaz.
    ControlList controls;
    controls.set(controls::FrameDurationLimits, Span<const int64_t>({33333, 33333}));

    DualStreamState state;

    // Requests oluştur (main+lores buffer eklenmiş)
    std::vector<std::unique_ptr<Request>> requests;
    const auto &mainBufs = allocator.buffers(mainStream);
    const auto &loresBufs = allocator.buffers(loresStream);
    size_t n = std::min(mainBufs.size(), loresBufs.size());
    if (n == 0) {
        std::cerr << "No buffers\n";
        return 1;
    }

    for (size_t i = 0; i < n; i++) {
        std::unique_ptr<Request> req = cam->createRequest();
        if (!req) {
            std::cerr << "createRequest failed\n";
            return 1;
        }

        req->controls() = controls;

        if (req->addBuffer(mainStream, mainBufs[i].get()) < 0 ||
            req->addBuffer(loresStream, loresBufs[i].get()) < 0) {
            std::cerr << "addBuffer failed\n";
            return 1;
        }
        requests.push_back(std::move(req));
    }

    // Callback: main+lores hazır olunca state’e koy
    cam->requestCompleted.connect([&](Request *req) {
        if (req->status() == Request::RequestCancelled) return;

        auto &bufMap = req->buffers();

        auto itMain = bufMap.find(mainStream);
        auto itLores = bufMap.find(loresStream);
        if (itMain == bufMap.end() || itLores == bufMap.end()) {
            req->reuse(Request::ReuseBuffers);
            cam->queueRequest(req);
            return;
        }

        FrameBuffer *fbMain = itMain->second;
        FrameBuffer *fbLores = itLores->second;

        // --- MAIN ---
        const auto &pMain = fbMain->planes()[0];
        void *memMain = mmap_plane(pMain);
        if (memMain) {
            cv::Mat bgr = yuv420_to_bgr(
                static_cast<uint8_t*>(memMain),
                mainCfg.size.width, mainCfg.size.height
            );
            munmap_plane(memMain, pMain);

            // flip (hflip + vflip)
            cv::flip(bgr, bgr, -1);

            std::lock_guard<std::mutex> lk(state.mtx);
            state.main_bgr = bgr;
            state.got_main = true;
        }

        // --- LORES (Y plane as gray) ---
        const auto &pLores = fbLores->planes()[0];
        void *memLores = mmap_plane(pLores);
        if (memLores) {
            cv::Mat gray = y_plane_as_gray_from_yuv420(
                static_cast<uint8_t*>(memLores),
                loresCfg.size.width, loresCfg.size.height
            );
            munmap_plane(memLores, pLores);

            // flip aynı olsun
            cv::flip(gray, gray, -1);

            std::lock_guard<std::mutex> lk(state.mtx);
            state.lores_gray = gray;
            state.got_lores = true;
        }

        {
            std::lock_guard<std::mutex> lk(state.mtx);
            if (state.got_main && state.got_lores) {
                state.cv.notify_one();
            }
        }

        req->reuse(Request::ReuseBuffers);
        cam->queueRequest(req);
    });

    if (cam->start()) {
        std::cerr << "Camera start failed\n";
        return 1;
    }
    for (auto &r : requests) {
        if (cam->queueRequest(r.get()) < 0) {
            std::cerr << "queueRequest failed\n";
            return 1;
        }
    }

    // -------- OpenCV QR part --------
    cv::QRCodeDetector detector;
    FpsCounter fps;

    cv::namedWindow("QR Code Detector", cv::WINDOW_AUTOSIZE);

    while (true) {
        cv::Mat mainFrame, gray;
        {
            std::unique_lock<std::mutex> lk(state.mtx);
            state.cv.wait(lk, [&] { return (state.got_main && state.got_lores) || !state.running; });
            if (!state.running) break;

            mainFrame = state.main_bgr.clone();
            gray = state.lores_gray.clone();
            state.got_main = false;
            state.got_lores = false;
        }

        // detectAndDecodeMulti
        std::vector<std::string> decoded_info;
        std::vector<std::vector<cv::Point2f>> points;
        bool ok = detector.detectAndDecodeMulti(gray, decoded_info, points);

        if (ok && !points.empty()) {
            // scaling main vs lores (Python ile aynı)
            double scale_x = static_cast<double>(mainFrame.cols) / gray.cols;
            double scale_y = static_cast<double>(mainFrame.rows) / gray.rows;

            for (size_t i = 0; i < decoded_info.size(); i++) {
                const std::string &data = decoded_info[i];
                if (i >= points.size() || points[i].size() < 4) continue;

                std::vector<cv::Point> pts(4);
                for (int j = 0; j < 4; j++) {
                    pts[j].x = static_cast<int>(points[i][j].x * scale_x);
                    pts[j].y = static_cast<int>(points[i][j].y * scale_y);
                }

                for (int j = 0; j < 4; j++) {
                    cv::line(mainFrame, pts[j], pts[(j + 1) % 4], cv::Scalar(255, 0, 0), 2);
                }

                if (!data.empty()) {
                    cv::putText(mainFrame, data, pts[0], cv::FONT_HERSHEY_COMPLEX, 1.0,
                                cv::Scalar(255, 255, 120), 2);
                    std::cout << "Data Found: " << data << std::endl;
                }
            }
        }

        fps.draw(mainFrame);
        cv::imshow("QR Code Detector", mainFrame);

        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q') break;
    }

    // cleanup
    state.running = false;
    cam->stop();
    cam->release();
    cm->stop();
    cv::destroyAllWindows();
    return 0;
}