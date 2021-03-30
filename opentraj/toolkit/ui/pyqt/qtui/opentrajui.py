# This Python file uses the following encoding: utf-8
import sys
import cv2
import numpy as np
# from PySide2.QtWidgets import QApplication, QCoreApplication, QMainWindow, QGraphicsView
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsItem, QGraphicsPixmapItem, QGraphicsPathItem, QGraphicsEllipseItem
from PyQt5.QtGui import QPixmap, QImage, QPainterPath, QBrush, QPen, QPainter, QColor
from PyQt5.QtCore import QEventLoop, QRectF
from .mainwindow import Ui_MainWindow


class GraphicsCircle(QGraphicsEllipseItem):
    pass

class GraphicsPath(QGraphicsPathItem):
    # def setBrush(self, Union, QBrush=None, QColor=None, Qt_GlobalColor=None, QGradient=None):
    def paint(self, painter: QPainter, styleOptionGraphicsItem, widget=None):
        # painter.setBrush()
        painter.drawPath(self.path())

    def boundingRect(self):
        penWidth = 1
        return QRectF(-10 - penWidth / 2, -10 - penWidth / 2, 20 + penWidth, 20 + penWidth)


class OpenTrajUI(QMainWindow, Ui_MainWindow):
    def __init__(self, reserve_n_agents=1):
        self.app = QApplication([])
        super(OpenTrajUI, self).__init__()
        self.setupUi(self)
        self.scene = QGraphicsScene()
#        self.scene.setSceneRect(QRectF(QPointF(-1000, 1000), QSizeF(2000, 2000)));

        self.graphicsView.setScene(self.scene)
        self.connectWidgets()
        self.show()
        self.pixmap_item = QGraphicsPixmapItem(QPixmap())
        self.scene.addItem(self.pixmap_item)

        custom_path = GraphicsPath()
        self.scene.addItem(custom_path)

        # graphic items
        self.circle_items = []
        self.circle_item_indicator = -1
        self.path_items = []
        self.path_item_indicator = -1
        self.pause = True

    def update_im(self, im_np):
        im_np = im_np[..., ::-1]  # convert BGR to RGB
        im_np = np.require(im_np, np.uint8, 'C')
        qimage = QImage(im_np.data, im_np.shape[1], im_np.shape[0], im_np.strides[0], QImage.Format_RGB888)
        self.pixmap_item.setPixmap(QPixmap(qimage))
#        self.pixmap_item.update()
#        self.scene.update()

    def __add_circle_item__(self):
        circle_item = QGraphicsEllipseItem()
        self.scene.addItem(circle_item)
        self.circle_items.append(circle_item)

    def __add_path_item__(self):
        path_item = QGraphicsPathItem()
        # path_item = GraphicsPath()

        self.scene.addItem(path_item)
        self.path_items.append(path_item)

    def erase_paths(self):
        for path_item in self.path_items:
            path_item.setPath(QPainterPath())
        self.path_item_indicator = -1

    def erase_circles(self):
        for circle_item in self.circle_items:
            circle_item.setRect(-100, -100, 0, 0)
        self.circle_item_indicator = -1

    def draw_path(self, path, color=[], width=[2]):
        if len(path) < 2:
            print('Warning: Path has less than 2 points')
            return

        self.path_item_indicator += 1
        if self.path_item_indicator >= len(self.path_items):
            self.__add_path_item__()
        painter_path = QPainterPath()
        painter_path.moveTo(path[0, 0], path[0, 1])
        for p in path:
            painter_path.lineTo(p[0], p[1])

        self.path_items[self.path_item_indicator].setPath(painter_path)

        pen = QPen(QColor(color[0], color[1], color[2]), width[0])
        self.path_items[self.path_item_indicator].setPen(pen)

    def draw_circle(self, center, radius, color, width):
        self.circle_item_indicator += 1
        if self.circle_item_indicator >= len(self.circle_items):
            self.__add_circle_item__()
        self.circle_items[self.circle_item_indicator].setRect(center[0] - radius, center[1] - radius,
                                                              radius * 2, radius * 2)
        pen = QPen(QColor(color[0], color[1], color[2]), width)
        self.circle_items[self.circle_item_indicator].setPen(pen)

    def processEvents(self):
        QApplication.processEvents(QEventLoop.AllEvents, 1)

    def connectWidgets(self):
        self.butPlay.clicked.connect(self.playPushed)
        # self.butVisualize.clicked.connect(self.changeTabToVisualize)
        # self.butTracking.clicked.connect(self.changeTabToTracking)
        # self.butAnalysis.clicked.connect(self.changeTabToAnalysis)

    def changeTabToVisualize(self):
        # if self.butVisualize.isChecked():
        #     return
        self.butVisualize.setChecked(True)
        self.butTracking.setChecked(False)
        self.butAnalysis.setChecked(False)

    def changeTabToTracking(self):
        # if self.butTracking.isChecked():
        #     return
        self.butTracking.setChecked(True)
        self.butVisualize.setChecked(False)
        self.butAnalysis.setChecked(False)

    def changeTabToAnalysis(self):
        # if self.butAnalysis.isChecked():
        #     return
        self.butAnalysis.setChecked(True)
        self.butVisualize.setChecked(False)
        self.butTracking.setChecked(False)

    def setTimestamp(self, timestamp):
        self.labelTimestamp.setText(str(timestamp))

    def playPushed(self):
        # test
        # self.textEdit.append("Python version:")
        # self.textEdit.append(sys.version)
        self.pause = not self.butPlay.isChecked()


if __name__ == "__main__":
    otui = OpenTrajUI()
    sys.exit(otui.app.exec_())
    QCoreApplication().processEvents()
