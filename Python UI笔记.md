# Python UI

## 第一个基础程序

```
import sys
import PyQt5.QtWidgets as qt
app = qt.QApplication(sys.argv)  #接受命令行参数
w = qt.QWidget()                 #创建窗体
w.setWindowTitle("标题")
w.show()           
app.exec()                       #循环接收消息
```

QtCore：包含核心非gui功能，和时间、文件、文件夹、数据、流、urls、mime类文件，进程，线程一起使用
qtgui：包含窗口系统，事件处理，2d图像，基本绘画，字体，文字类
qtwidge：包含创建桌面元素控件

## 控件

1. 按钮(QPushButton)
   ```
   bt = qt.QPushButton("按钮",w) #设置按钮从属于w窗体
   ```

2. 文本(QLabel)
   ```
   lab = qt.QLabel("账号",w)
   lab.setGeometry(30,30,80,80) //设置位置和宽高
   ```

3. 输入框(QLineEdit)

   ```
   edit = qt.QLineEdit(w)
   edit.setPlaceholderText("账号")
   ```

4.  调整窗口

   ```
   w.resize(300,300) //设置窗口大小
   w.move(0,0) //移动窗口
   ```

5. 设置图标

   ```
   w.setWindowIcon(QIcon(""))
   ```

## 布局

1. 盒子布局(QBoxLayout)
   QHBoxLayout 和 QVBoxLayout 负责水平和垂直布局

   ```
   layout = QVBoxLayout()
   bt = qt.QPushButton("按钮",w)
   layout.addWidget(bt) #将按钮加入布局器
   layout.addStretch() #设置一个伸缩器
   setLayout(layout) #设置使用layout布局器
   
   
   程序二 
   
   
   import sys
   from PyQt5.QtWidgets import *
   class Mywindows(QWidget):
       def __init__(self):
           super().__init__()
           self.init_ui()
       def init_ui(self):
           container = QVBoxLayout()
           hobby_box = QGroupBox("爱好")
           v_layout = QVBoxLayout()
           bt1 = QRadioButton("1")
           bt2 = QRadioButton("2")
           bt3 = QRadioButton("3")
           v_layout.addWidget(bt1)
           v_layout.addWidget(bt2)
           v_layout.addWidget(bt3)
           hobby_box.setLayout(v_layout)
           gender = QGroupBox("性别")
           h_layout = QHBoxLayout()
           bt4 = QRadioButton("4")
           bt5 = QRadioButton("5")
           h_layout.addWidget(bt4)
           h_layout.addWidget(bt5)
           gender.setLayout(h_layout)
           container.addWidget(hobby_box)
           container.addWidget(gender)
           self.setLayout(container)
   app = QApplication(sys.argv)
   w = Mywindows()
   w.show()
   app.exec()
   ```

2. 网格布局(QGridLayout)
   注：布局器里可以嵌套布局器，用  layout.addLayout(grid)来添加到当前的布局器，嵌套的布局器和正常的布局器一样使用，每创建一个控件用addWidget添加到布局器

   ```
   import sys
   from PyQt5.QtWidgets import *
   class Mywindows(QWidget):
       def __init__(self):
           super().__init__()
           self.init_ui()
       def init_ui(self):
           data = {
               0: ["7", "8", "9", "+", "("],
               1: ["4", "5", "6", "-", ")"],
               2: ["1", "2", "3", "*", "<-"],
               3: ["0", ".", "=", "/", "c"]
           }
           layout = QVBoxLayout()
           edit = QLineEdit()
           edit.setPlaceholderText("请输入内容")
           layout.addWidget(edit)
           grid = QGridLayout()
           for line,content1 in data.items():
               for clon,content2 in enumerate(content1):
                   button = QPushButton(content2)
                   grid.addWidget(button,line,clon)
           layout.addLayout(grid)
           self.setLayout(layout)
   
   
   
   app = QApplication(sys.argv)
   w = Mywindows()
   w.resize(400,400)
   w.move(0,0)
   w.show()
   app.exec()
   ```

3. 表单布局(QFormLayout)

4. 抽屉布局(QstackedLayout)
   提供了多页面切换，一次只能看到一个界面，结合消息控件可以实现一些东西

   ```
   import sys
   from PyQt5.QtWidgets import *
   class windows1(QWidget):
       def __init__(self):
           super().__init__()
           QLabel("窗口1",self)
           self.setStyleSheet("background-color:green;")
   class windows2(QWidget):
       def __init__(self):
           super().__init__()
           QLabel("窗口二",self)
   class Mywindows(QWidget):
       def __init__(self):
           super().__init__()
           self.init_ui()
       def win1_click(self):
           self.stlayout.setCurrentIndex(0)
       def win2_click(self):
           self.stlayout.setCurrentIndex(1)
       def init_ui(self):
           container = QVBoxLayout()
           self.stlayout = QStackedLayout()
           win1 = windows1()
           win2 = windows2()
           ww = QWidget()
           self.stlayout.addWidget(win1)
           self.stlayout.addWidget(win2)
           ww.setLayout(self.stlayout)
           btn = QPushButton("按钮1")
           btn2 = QPushButton("按钮2")
           btn.clicked.connect(self.win1_click)
           btn2.clicked.connect(self.win2_click)
           container.addWidget(ww)
           container.addWidget(btn)
           container.addWidget(btn2)
           self.setLayout(container)
   app = QApplication(sys.argv)
   w = Mywindows()
   w.resize(400,400)
   w.move(0,0)
   w.show()
   app.exec()
   ```

   QWidget 没有划分菜单、工具栏、状态栏、主窗口

   QMainWindows 包含菜单栏、工具栏、状态栏、中间部分为主窗口区域

   QDialog 对话框窗口的基类
   lab.setWordWrap(True)自动换行
   lab.setAlignment(Qt.AlignTop)设置向上对齐
   score = QScrollArea()设置滚动对象
   score.setWidget(self.lab) 滚动
   v_layout = QVBoxLayout()
   v_laytou.addWidget(score)

   先把label添加到滚动对象里，再把滚动对象添加到垂直布局器里，再把垂直布局器放在主布局器里
   往lab框里添加东西 
   self.lab.setText("1")
   self.lab.resize(440,self.lab.frameSize().height()+15)

   self.lab.repaint()#更新内容，否则lab框里的东西不变

5. 菜单
   ```
   import sys
   from PyQt5.QtWidgets import *
   class Mywindows(QMainWindow):
       def __init__(self):
           super().__init__()
           self.init_ui()
       def init_ui(self):
           menu = self.menuBar()
           file_menu = menu.addMenu("菜单")
           file_menu.addAction("b")
           content_menu = menu.addMenu("内容")
           content_menu.addAction("c")
           content_menu.addAction("d")
   
   
   app = QApplication(sys.argv)
   w = Mywindows()
   w.resize(400, 400)
   w.move(0, 0)
   w.show()
   app.exec()
   
   ```

## 信号与槽

1. 信号与槽绑定：对象.事件.connect(槽函数)
   ```python
   btn.clicked.connect(self.win1_click)
   my_signal = pyqtSignal(str) #自定义信号，必须写在类属性位置
   self.my_signal.connect(self.win1_click) #绑定槽函数和自定义信号
   self.my_signal.emit() #激发信号，调用槽函数
   
   ```

   

## QtDesigner使用

- 里面的控件和mfc一样，可以拖动使用

- form里面的preview可以预览

- 导入到python里
  ```
  import sys
  from PyQt5.QtWidgets import *
  from PyQt5 import uic
  app = QApplication(sys.argv)
  ui = uic.loadUi("./untitled.ui")
  ui.show()
  app.exec()
  ```

- 打印designer设计的界面，里面每个控件的属性

  ```
  ui = uic.loadUi("./login.ui")
  print(ui) #UI文件中对顶层的对象的所有属性
  print(ui.__dict__) #每个控件的名字
  print(ui.label.text())#label控件的text属性的值
  #绑定按钮和信号
  login_button = self.ui.pushButton
  login_button.clicked.connect(self.click_login)
  
  案例
  import sys
  from PyQt5.QtWidgets import *
  from PyQt5 import uic
  
  
  class mywindow(QWidget):
      def __init__(self):
          super().__init__()
          self.init()
  
      def init(self):
          self.ui = uic.loadUi("login.ui")
          login_button = self.ui.pushButton
          login_button.clicked.connect(self.click_login)
      def click_login(self):
          self.ui.textBrowser.setText(f"{self.ui.lineEdit.text()}欢迎登录")
          self.ui.textBrowser.repaint()
  app = QApplication(sys.argv)
  w = mywindow()
  w.ui.show()
  app.exec()
  
  ```



## 多线程

- 使用多线程的原因：如果click以后执行了某个耗时的循环，则界面会卡在那里，点不动东西，输不了东西，此时换成多线程，在click以后并没有在主线程执行耗时的循环，而是开辟一个子线程，用子线程去跑槽函数，则界面不会卡主

- ```
  import sys
  import time
  
  from PyQt5.QtCore import QThread
  from PyQt5.QtWidgets import *
  from PyQt5 import uic
  
  
  class Mythread(QThread):
      def __init__(self):
          super().__init__()
  
      def run(self):
          for i in range(5):
              print(f"{i}")
          time.sleep(1)
  
  class mywindow(QWidget):
      def __init__(self):
          super().__init__()
          self.init()
  
      def init(self):
          self.ui = uic.loadUi("untitled.ui")
          button = self.ui.pushButton
          button.clicked.connect(self.ka)
  
      def buka(self):
          self.thread = Mythread()
          self.thread.start()
  
      def ka(self):
          for i in range(5):
              print(f"{i}")
              time.sleep(1)
  
  
  app = QApplication(sys.argv)
  w = mywindow()
  w.ui.show()
  app.exec()
  
  ```

- 

