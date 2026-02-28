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

// Helper to track FPS
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

// Represents a mapped buffer
struct MappedBuffer {
    void *memory;
    size_t size;

    MappedBuffer(const FrameBuffer::Plane &plane) {
        size = plane.length;
        memory = mmap(nullptr, size, PROT_READ, MAP_SHARED, plane.fd.get(), 0);
    }

    ~MappedBuffer() {
        if (memory != MAP_FAILED)
            munmap(memory, size);
    }

    bool isValid() const { return memory != MAP_FAILED; }
};

struct AppState {
    cv::Mat main_frame;
    cv::Mat lores_gray;
    std::mutex mtx;
    std::condition_variable cv;
    bool fresh = false;
    std::atomic<bool> running{true};
};

int main() {
    std::unique_ptr<CameraManager> cm = std::make_unique<CameraManager>();
    if (cm->start()) return 1;

    if (cm->cameras().empty()) {
        std::cerr << "No cameras found" << std::endl;
        return 1;
    }

    std::shared_ptr<Camera> camera = cm->cameras()[0];
    camera->acquire();

    std::unique_ptr<CameraConfiguration> config = camera->generateConfiguration({ StreamRole::Viewfinder, StreamRole::Viewfinder });
    
    // Main stream for display (XRGB8888 is widely supported and easy to convert to BGR)
    StreamConfiguration &mainCfg = config->at(0);
    mainCfg.size.width = 320;
    mainCfg.size.height = 240;
    mainCfg.pixelFormat = formats::XRGB8888;

    // Lores stream for analysis (YUV420 to get Greyscale Y plane)
    StreamConfiguration &loresCfg = config->at(1);
    loresCfg.size.width = 320;
    loresCfg.size.height = 240;
    loresCfg.pixelFormat = formats::YUV420;

    config->validate();
    camera->configure(config.get());

    FrameBufferAllocator allocator(camera);
    for (StreamConfiguration &cfg : *config) {
        allocator.allocate(cfg.stream());
    }

    Stream *mainStream = mainCfg.stream();
    Stream *loresStream = loresCfg.stream();

    AppState state;
    std::vector<std::unique_ptr<Request>> requests;

    const std::vector<std::unique_ptr<FrameBuffer>> &mainBuffers = allocator.buffers(mainStream);
    const std::vector<std::unique_ptr<FrameBuffer>> &loresBuffers = allocator.buffers(loresStream);

    for (unsigned int i = 0; i < mainBuffers.size(); ++i) {
        std::unique_ptr<Request> request = camera->createRequest();
        request->addBuffer(mainStream, mainBuffers[i].get());
        request->addBuffer(loresStream, loresBuffers[i].get());
        requests.push_back(std::move(request));
    }

    // Some libcamera versions require a context object for Signal::connect
    camera->requestCompleted.connect(camera.get(), [&](Request *request) {
        if (request->status() == Request::RequestCancelled) return;

        FrameBuffer *mainFb = request->buffers().at(mainStream);
        FrameBuffer *loresFb = request->buffers().at(loresStream);

        // Map main buffer (XRGB8888)
        MappedBuffer mainMap(mainFb->planes()[0]);
        // Map lores buffer (YUV420, we only need the first plane which is Y / Gray)
        MappedBuffer loresMap(loresFb->planes()[0]);

        if (mainMap.isValid() && loresMap.isValid()) {
            cv::Mat xrgb(240, 320, CV_8UC4, mainMap.memory);
            cv::Mat bgr;
            cv::cvtColor(xrgb, bgr, cv::COLOR_BGRA2BGR); // XRGB is effectively BGRA in OpenCV terms usually
            
            // Apply flips as in Python: transform=Transform(hflip=True, vflip=True)
            cv::flip(bgr, bgr, -1);

            cv::Mat gray(240, 320, CV_8UC1, loresMap.memory);
            cv::Mat gray_clone = gray.clone();
            cv::flip(gray_clone, gray_clone, -1);

            {
                std::lock_guard<std::mutex> lock(state.mtx);
                state.main_frame = bgr.clone();
                state.lores_gray = gray_clone;
                state.fresh = true;
            }
            state.cv.notify_one();
        }

        request->reuse(Request::ReuseBuffers);
        camera->queueRequest(request);
    });

    camera->start();

    for (auto &request : requests) camera->queueRequest(request.get());

    cv::QRCodeDetector detector;
    FpsCounter fps;

    while (state.running) {
        cv::Mat display, gray;
        {
            std::unique_lock<std::mutex> lock(state.mtx);
            state.cv.wait(lock, [&] { return state.fresh || !state.running; });
            if (!state.running) break;
            display = state.main_frame.clone();
            gray = state.lores_gray.clone();
            state.fresh = false;
        }

        std::vector<std::string> decoded_info;
        std::vector<cv::Point2f> points;
        bool ok = detector.detectAndDecodeMulti(gray, decoded_info, points);

        if (ok && !points.empty()) {
            double scale_x = (double)display.cols / gray.cols;
            double scale_y = (double)display.rows / gray.rows;

            for (size_t i = 0; i < decoded_info.size(); i++) {
                std::string data = decoded_info[i];
                // In C++ it returns a flat vector of points or vector<vector<Point2f>>
                // detectAndDecodeMulti(InputArray img, std::vector<std::string>& decoded_info, OutputArray points...)
                // points is OutputArray. If it's std::vector<cv::Point2f>, it's flat.
                
                if (points.size() >= (i + 1) * 4) {
                    std::vector<cv::Point> pts;
                    for (int j = 0; j < 4; j++) {
                        cv::Point2f p = points[i * 4 + j];
                        pts.push_back(cv::Point(p.x * scale_x, p.y * scale_y));
                    }

                    for (int j = 0; j < 4; j++) {
                        cv::line(display, pts[j], pts[(j + 1) % 4], cv::Scalar(255, 0, 0), 2);
                    }

                    if (!data.empty()) {
                        cv::putText(display, data, pts[0], cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(255, 255, 120), 2);
                        std::cout << "Data Found: " << data << std::endl;
                    }
                }
            }
        }

        fps.draw(display);
        cv::imshow("QR Code Detector", display);
        if (cv::waitKey(1) == 'q') state.running = false;
    }

    state.running = false;
    state.cv.notify_all();

    camera->stop();
    camera->release();
    cm->stop();

    return 0;
}