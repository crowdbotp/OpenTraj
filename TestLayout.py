# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'TestLayout.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#


from PyQt4 import QtCore, QtGui
import json 

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(652, 588)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.gridLayout = QtGui.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.horizontalLayout_4 = QtGui.QHBoxLayout()
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem)
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout_4.addWidget(self.label)
        self.comboBox = QtGui.QComboBox(self.centralwidget)
        self.comboBox.setObjectName(_fromUtf8("comboBox"))
        self.horizontalLayout_4.addWidget(self.comboBox)
        self.pushButton = QtGui.QPushButton(self.centralwidget)
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.horizontalLayout_4.addWidget(self.pushButton)
        self.gridLayout.addLayout(self.horizontalLayout_4, 0, 0, 1, 1)
        self.tabWidget = QtGui.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(_fromUtf8("tabWidget"))
        self.tab = QtGui.QWidget()
        self.tab.setObjectName(_fromUtf8("tab"))
        self.layoutWidget = QtGui.QWidget(self.tab)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 90, 301, 41))
        self.layoutWidget.setObjectName(_fromUtf8("layoutWidget"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.label_4 = QtGui.QLabel(self.layoutWidget)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.horizontalLayout_2.addWidget(self.label_4)
        self.textBrowser_2 = QtGui.QTextBrowser(self.layoutWidget)
        self.textBrowser_2.setObjectName(_fromUtf8("textBrowser_2"))
        self.horizontalLayout_2.addWidget(self.textBrowser_2)
        self.layoutWidget_3 = QtGui.QWidget(self.tab)
        self.layoutWidget_3.setGeometry(QtCore.QRect(10, 210, 301, 111))
        self.layoutWidget_3.setObjectName(_fromUtf8("layoutWidget_3"))
        self.horizontalLayout_5 = QtGui.QHBoxLayout(self.layoutWidget_3)
        self.horizontalLayout_5.setObjectName(_fromUtf8("horizontalLayout_5"))
        self.label_6 = QtGui.QLabel(self.layoutWidget_3)
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.horizontalLayout_5.addWidget(self.label_6)
        self.textBrowser_4 = QtGui.QTextBrowser(self.layoutWidget_3)
        self.textBrowser_4.setObjectName(_fromUtf8("textBrowser_4"))
        self.horizontalLayout_5.addWidget(self.textBrowser_4)
        self.layoutWidget_4 = QtGui.QWidget(self.tab)
        self.layoutWidget_4.setGeometry(QtCore.QRect(10, 150, 301, 41))
        self.layoutWidget_4.setObjectName(_fromUtf8("layoutWidget_4"))
        self.horizontalLayout_6 = QtGui.QHBoxLayout(self.layoutWidget_4)
        self.horizontalLayout_6.setObjectName(_fromUtf8("horizontalLayout_6"))
        self.label_7 = QtGui.QLabel(self.layoutWidget_4)
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.horizontalLayout_6.addWidget(self.label_7)
        spacerItem1 = QtGui.QSpacerItem(85, 20, QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem1)
        self.textBrowser_5 = QtGui.QTextBrowser(self.layoutWidget_4)
        self.textBrowser_5.setOpenExternalLinks(True)
        self.textBrowser_5.setObjectName(_fromUtf8("textBrowser_5"))
        self.horizontalLayout_6.addWidget(self.textBrowser_5)
        self.widget = QtGui.QWidget(self.tab)
        self.widget.setGeometry(QtCore.QRect(10, 40, 301, 31))
        self.widget.setObjectName(_fromUtf8("widget"))
        self.horizontalLayout = QtGui.QHBoxLayout(self.widget)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.label_3 = QtGui.QLabel(self.widget)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.horizontalLayout.addWidget(self.label_3)
        self.textBrowser = QtGui.QTextBrowser(self.widget)
        self.textBrowser.setObjectName(_fromUtf8("textBrowser"))
        self.horizontalLayout.addWidget(self.textBrowser)
        self.tabWidget.addTab(self.tab, _fromUtf8(""))
        self.tab_2 = QtGui.QWidget()
        self.tab_2.setObjectName(_fromUtf8("tab_2"))
        self.gridLayout_2 = QtGui.QGridLayout(self.tab_2)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.label_2 = QtGui.QLabel(self.tab_2)
        self.label_2.setFrameShape(QtGui.QFrame.Box)
        self.label_2.setText(_fromUtf8(""))
        self.label_2.setPixmap(QtGui.QPixmap(_fromUtf8("reference_images/Pets2009.png")))
        self.label_2.setScaledContents(True)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout_2.addWidget(self.label_2, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tab_2, _fromUtf8(""))
        self.tab_3 = QtGui.QWidget()
        self.tab_3.setObjectName(_fromUtf8("tab_3"))
        self.label_5 = QtGui.QLabel(self.tab_3)
        self.label_5.setGeometry(QtCore.QRect(50, 50, 71, 21))
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.label_8 = QtGui.QLabel(self.tab_3)
        self.label_8.setGeometry(QtCore.QRect(50, 90, 81, 21))
        self.label_8.setObjectName(_fromUtf8("label_8"))
        self.label_9 = QtGui.QLabel(self.tab_3)
        self.label_9.setGeometry(QtCore.QRect(140, 50, 67, 17))
        self.label_9.setOpenExternalLinks(True)
        self.label_9.setObjectName(_fromUtf8("label_9"))
        self.label_10 = QtGui.QLabel(self.tab_3)
        self.label_10.setGeometry(QtCore.QRect(140, 90, 67, 17))
        self.label_10.setOpenExternalLinks(True)
        self.label_10.setObjectName(_fromUtf8("label_10"))
        self.tabWidget.addTab(self.tab_3, _fromUtf8(""))
        self.gridLayout.addWidget(self.tabWidget, 2, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 652, 22))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.populateDropdown()
        self.home()
    
    def populateDropdown(self):
        self.data = json.loads(open('./config.json').read())
        dataset_list  = [str(datasets['name']) for datasets in self.data['Datasets'] ]
        self.comboBox.addItems(dataset_list)
        self.datasets_Dict = { k:v for (k,v) in zip(dataset_list, range(len(dataset_list)))}

    def home(self):
        self.pushButton.clicked.connect(self.populateFields)
    
    def populateFields(self):
        selected_dataset = self.comboBox.currentText()
        dataset = self.data['Datasets'][self.datasets_Dict[str(selected_dataset)]]
        self.textBrowser.setText(dataset['name']) 
        self.textBrowser.adjustSize()
        self.textBrowser_4.setText(dataset['description'])
        # self.textBrowser_4.adjustSize()
        self.label_9.setText('<a href="'+str(dataset['download_annotation'])+'"style="text-decoration: none">Only annotations'+str(dataset['detail_1'])+'<b>&#8595;</b>'+'</a>')
        self.label_9.adjustSize()
        self.label_10.setText('<a href="'+str(dataset['download_annotation'])+'"style="text-decoration: none">Annotations plus video'+str(dataset['detail_2'])+'<b>&#8595;</b>'+'</a>')
        self.label_10.adjustSize()
        self.textBrowser_5.setText('<a href="'+dataset['paper']+'">Link</a>')
        self.label_2.setPixmap(QtGui.QPixmap(_fromUtf8(dataset['ref_image'])))


    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.label.setText(_translate("MainWindow", "Select Dataset", None))
        self.pushButton.setText(_translate("MainWindow", "Select", None))
        self.label_4.setText(_translate("MainWindow", "Provided By", None))
        self.label_6.setText(_translate("MainWindow", "Description", None))
        self.label_7.setText(_translate("MainWindow", "Paper", None))
        self.label_3.setText(_translate("MainWindow", "Dataset Name", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Summary", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Visualization", None))
        self.label_5.setText(_translate("MainWindow", "Download", None))
        self.label_8.setText(_translate("MainWindow", "Download", None))
        self.label_9.setText(_translate("MainWindow", " ", None))
        self.label_10.setText(_translate("MainWindow", " ", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "Download", None))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

