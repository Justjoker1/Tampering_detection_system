import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QFileDialog, QLineEdit
)
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl, QThread, pyqtSignal, QSize
from features_extractor.features_extractor import extract_features
from stage_one import stage_one_predict
from stage_two import stage_two_predict
from PyQt5.QtWidgets import QSizePolicy

class FeatureExtractionThread(QThread):
    extraction_finished = pyqtSignal(str)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path

    def run(self):
        result_path = r"./output/extracted_video/result_demo.mp4"
        try:
            extract_features(self.video_path, result_path=result_path, show_plot=False, show_text=False, show_video=False, save_video=True)
            self.extraction_finished.emit(result_path)
        except Exception as e:
            print(f"特征提取错误：{e}")
            self.extraction_finished.emit(None)

class DetectionThread(QThread):
    detection_finished = pyqtSignal(int, int)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path

    def run(self):
        try:
            # 调用一阶段检测
            input = [
                '--data_dir', "./output/tobedetected_video/",
                '--weights', "./checkpoints/stageone_model_best.pth",
            ]
            tamper_one = stage_one_predict.main(input_args=input)
            if tamper_one == 1:
                self.detection_finished.emit(1, 0)
            else:
                # 调用二阶段检测
                tamper_two = stage_two_predict.main()
                self.detection_finished.emit(0, tamper_two)
        except Exception as e:
            print(f"检测错误：{e}")
            self.detection_finished.emit(-1, -1)


class VideoApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('视频应用程序')
        self.setGeometry(100, 100, 800, 600)
        self.resize_thread = None
        self.feature_extraction_thread = None
        self.detection_thread = None
        self.processed_video_path = None

        # 创建中心部件
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # 创建布局
        main_layout = QVBoxLayout(central_widget)

        # 视频显示区域
        video_layout = QHBoxLayout()
        self.original_video_widget = QVideoWidget()
        self.detected_video_widget = QVideoWidget()
        self.original_video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.detected_video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        video_layout.addWidget(self.original_video_widget, 1)
        video_layout.addWidget(self.detected_video_widget, 1)
        main_layout.addLayout(video_layout)

        # 视频路径输入
        self.video_path_input = QLineEdit()
        self.video_path_input.setReadOnly(True)
        main_layout.addWidget(self.video_path_input)

        self.select_video_button = QPushButton('选择附件', self)
        self.select_video_button.clicked.connect(self.select_video)
        main_layout.addWidget(self.select_video_button)

        # 控制按钮
        button_layout = QHBoxLayout()
        self.play_button = QPushButton('播放')
        self.play_button.clicked.connect(self.play_video)
        self.pause_button = QPushButton('暂停')
        self.pause_button.clicked.connect(self.pause_video)
        self.detect_button = QPushButton('检测')
        self.detect_button.clicked.connect(self.detect_video)
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.detect_button)
        main_layout.addLayout(button_layout)

        # 状态显示区域
        text_layout = QHBoxLayout()
        # 左侧文本框（过程状态输出）
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.append('<b style="color: red;">=== 判定条件输出 ===</b>')
        self.status_text.append("请输入视频，交通特征提取将自动进行")
        text_layout.addWidget(self.status_text)

        # 右侧文本框（判定条件输出）
        self.condition_text = QTextEdit()
        self.condition_text.setReadOnly(True)
        self.condition_text.append('<b style="color: red;">=== 判定条件输出 ===</b>')
        text_layout.addWidget(self.condition_text)

        main_layout.addLayout(text_layout)

        # 初始化时禁用播放和暂停按钮
        self.play_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.detect_button.setEnabled(False)


    def select_video(self):
        video_path, _ = QFileDialog.getOpenFileName(
            self, '选择附件', '', '视频文件 (*.mp4 *.avi *.mov)'
        )

        if video_path:
            import shutil
            target_folder = r"./output/tobedetected_video/"
            target_path = os.path.join(target_folder, "tobedetected_video.mp4")
            shutil.copy2(video_path, target_path)
            self.video_path_input.setText(video_path)
            self.start_feature_extraction(video_path)


    def start_feature_extraction(self, video_path):
        if self.feature_extraction_thread and self.feature_extraction_thread.isRunning():
            self.feature_extraction_thread.terminate()

        self.feature_extraction_thread = FeatureExtractionThread(video_path)
        self.feature_extraction_thread.extraction_finished.connect(self.on_extraction_complete)
        self.feature_extraction_thread.start()
        self.status_text.append("特征提取中，请等待...")
        self.play_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.detect_button.setEnabled(False)

    def on_extraction_complete(self, result_path):
        if result_path and os.path.exists(result_path):
            self.processed_video_path = result_path
            media = QMediaContent(QUrl.fromLocalFile(result_path))
            self.detected_media_player = QMediaPlayer()
            self.detected_media_player.setVideoOutput(self.detected_video_widget)
            self.detected_media_player.setMedia(media)

            original_media = QMediaContent(QUrl.fromLocalFile(self.video_path_input.text()))
            self.original_media_player = QMediaPlayer()
            self.original_media_player.setVideoOutput(self.original_video_widget)
            self.original_media_player.setMedia(original_media)

            self.status_text.append("特征提取完成，可以播放和检测视频")
            # 按钮可以继续保留，以便用户手动暂停/重新播放
            self.play_button.setEnabled(True)
            self.pause_button.setEnabled(True)
            self.detect_button.setEnabled(True)
        else:
            self.status_text.append("特征提取失败，请检查文件格式")

    def play_video(self):
        if hasattr(self, 'original_media_player'):
            self.original_media_player.play()
        if hasattr(self, 'detected_media_player'):
            self.detected_media_player.play()

    def pause_video(self):
        if hasattr(self, 'original_media_player'):
            self.original_media_player.pause()
        if hasattr(self, 'detected_media_player'):
            self.detected_media_player.pause()

    def detect_video(self):
        if self.video_path_input.text():
            self.status_text.append("视频检测中，请等待...")
            if self.detection_thread and self.detection_thread.isRunning():
                self.detection_thread.terminate()
            self.detection_thread = DetectionThread(self.video_path_input.text())
            self.detection_thread.detection_finished.connect(self.on_detection_complete)
            self.detection_thread.start()

    def on_detection_complete(self, tamperone, tampertwo):
        self.condition_text.append("检测结果输出：")
        if tamperone == 1 and tampertwo == 0:
            self.condition_text.append("第一阶段检测为篡改")
        elif tampertwo == 1 and tamperone == 0:
            self.condition_text.append("第二阶段检测为篡改")
        elif tamperone == 0 and tampertwo == 0:
            self.condition_text.append("两阶段均未检测到篡改")
        else:
            self.condition_text.append("检测出错，请检查文件格式或模型文件")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    video_app = VideoApp()
    video_app.show()
    sys.exit(app.exec_())