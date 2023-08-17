# C++ Qt学习笔记

很多语法都和python的pyqt5一样，不在此多赘述，这里主要写结构和一些不同与pyqt5的语法，简单语法到时候自己查文档

C++写qt的结构大概是：

一个创建项目开始就有的mainWindow.cpp、main.cpp、mainwindow.h、项目名.pro

main.cpp只做一件事，创建application，创建窗口类，展示窗口，运行程序，其他内容不要放在main.cpp里

mainwindow.cpp：所有的界面设计都写在这里面，整个界面显示的内容都写在这儿，信号和槽函数的链接，一些函数的实现写在这

mainwindow.h：一般变量的声明和函数的声明写在这里

如果要定义槽函数，一般是创建一个类，类里面有槽函数这个方法，一般类继承QObject，创建的时候会同时创建cpp文件和.h文件，自定义信号的声明和一些函数方法的声明写在头文件里，在cpp文件里实现函数方法

设计界面的时候，要用new，不要用类名创建对象

QDebug输出QString带引号，解决办法将Qstring转换为char *类型，变量名.toUtf8().data()

连接信号connect

断开信号disconnect

自定义信号
```
创建信号
在signals下写信号，信号写成函数形式
signals:
void hungery();
void hungery(QString foodname);

槽函数
在public下进行声明、在其他地方定义、或者声明的时候直接定义

绑定函数，分为有函数重载和无函数重载情况
无函数重载情况
直接connect(tea,&teacher::hungery,stu,&student::treat);即可
有函数重载情况需要先声明一个带参数的函数指针，指向函数，再指针填到信号和槽函数拦里
    void (teacher ::*pTeacher)(QString) = &teacher::hungery;
    void (student ::*pStudent)(QString) = &student::treat;
    connect(tea,pTeacher,stu,pStudent);
```
```
teacher里有一个自定义信号hungery和重载后的hungery(QString foodname)
student有一个槽函数，treat和重载后的treat(QString foodname)
将hungery信号和treat方法连接，有一个函数emit自定义信号


main.cpp
#include "widge.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    Widge w;
    w.show();
    return a.exec();
}


student.h
#ifndef STUDENT_H
#define STUDENT_H
#include <QObject>
class student : public QObject
{
    Q_OBJECT
public:
    explicit student(QObject *parent = nullptr);
    void treat();
    void treat(QString foodname);
signals:
};

#endif // STUDENT_H


student.cpp
#include "student.h"
#include<QtDebug>
student::student(QObject *parent) : QObject(parent)
{

}
void student::treat()
{
    qDebug()<<"请客吃饭";
}
void student::treat(QString foodname)
{
    qDebug()<<"请吃"<<foodname;
}


teacher.h
#ifndef TEACHER_H
#define TEACHER_H
#include <QObject>
class teacher : public QObject
{
    Q_OBJECT
public:
    explicit teacher(QObject *parent = nullptr);
signals:
void hungery();
void hungery(QString foodname);
};

#endif // TEACHER_H


teacher.cpp
#include "teacher.h"
teacher::teacher(QObject *parent) : QObject(parent)
{

}


widge.h
#ifndef WIDGE_H
#define WIDGE_H

#include <QMainWindow>
#include "teacher.h"
#include "student.h"

QT_BEGIN_NAMESPACE
namespace Ui { class Widge; }
QT_END_NAMESPACE

class Widge : public QMainWindow
{
    Q_OBJECT

public:
    Widge(QWidget *parent = nullptr);
    ~Widge();

private:
    Ui::Widge *ui;
    teacher * tea;
    student * stu;
    void classisover();
};
#endif // WIDGE_H


widge.cpp
#include "widge.h"
#include "ui_widge.h"

Widge::Widge(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::Widge)
{
    //ui->setupUi(this);
    this->tea = new teacher(this);
    this -> stu = new student(this);
    //connect(tea,&teacher::hungery,stu,&student::treat);
    void (teacher ::*pTeacher)(QString) = &teacher::hungery;
    void (student ::*pStudent)(QString) = &student::treat;
    connect(tea,pTeacher,stu,pStudent);
    classisover();
}
void Widge::classisover()
{
    emit tea->hungery();
}
Widge::~Widge()
{
    delete ui;
}

```

connect的重载

```
connect(btn1,&QPushButton::clicked,[=]()
{
Qdialog *dlg= new Qdialog("窗口1");
dlg->exec();
}
);
```

自定义控件

作用：直接把封装好的控件集合到一个ui中

创建一个qt设计师类，把功能balabala的都写好，作为一个封装好的小组件
在主界面，用widge腾出一块空间给小组件，右击widget，选择提升，提升的类名称写那个小组件的名称，选择添加，选上全局，再编译就添加好了



鼠标事件

捕获鼠标事件需要修改控件的代码，所以需要先创建一个控件，再修改创建的控件的代码，再在主窗口提升写好的控件，即可捕获

```
例：
创建一个类，继承于qwidget
在创建的.h中添加
    void enterEvent(QEvent *event);
    void leaveEvent(QEvent *);
在创建的cpp文件中
void my::enterEvent(QEvent *event)
{
    qDebug() <<"进入";
}
void my::leaveEvent(QEvent *)
{
    qDebug()<<"离开";
}

在主项目中搞一个widget控件，提升控件，即可，在提升控件的时候，选择的控件要和创建的类继承自同一个类，提升的时候选择类要选择对

其他鼠标事件还有mousePressEvent等，直接查文档

按下左键开始移动（防止中途换右键），不能在移动的代码中判断左键是否按下，即ev==qt::leftbutton 而是要用buttons & Qt::leftbutton

如果设置鼠标追踪，则什么事情都不发生也会捕获鼠标事件
```

定时器

```
定时器时间 timeevent
创建一个控件（比如一个label），在mainwindow头文件中声明重写timeevent事件
在mainwindow.cpp文件中重写
void mainwindow::timeevent(Qtimeevent*)
{
static int num = 1;
ui->label->settext(Qstring::number(num++));
}
启动定时器
starttimer(1000);每秒执行一次

启动定时器会有一个int返回值，作用是区分定时器，比如timeevent里有两个定时器，则需要区分，写一个成员变量来接收返回值
id1 = starttimer(1000)
id2 = starttimer(2000)

void mainwindow::timeevent(Qtimeevent*ev)
if(ev->timeId == id)
{
static int num = 1;
ui->label->settext(Qstring::number(num++));
}
if(ev->timeId == id2)
{
static int num = 1;
ui->label2->settext(Qstring::number(num++));
}
}

第二种定时器
包含Qtimer
QTimer *timer = new QTimer(this);
timer->start(500); 每0.5秒发个timeout信号
connect(timer,&QTimer::timeout,[=]()
{

});
```

事件

```
事件通过事件分发器来指派不同的任务
用户点击，信号到事件分发器，事件分发器再指派任务，事件分发器
bool event 如果返回true则不往下分发事件，可以在此进行事件的拦截，对于其他不需要拦截的事件，需要
else
{
return QLabel::event(e);
}

在事件到事件分发器之前，可以拦截，使用时间过滤器eventfilter
分两步，安装事件过滤器，重写事件过滤器
```

绘画

```
qt中绘画依靠绘画事件
void PaintEvent(QPainterEvent *event)
先在头文件中声明重写，在cpp文件中重写
void MainWindow::paintEvent(QPaintEvent *event)
{
QPainter painter(this); //实例化画家对象
painter.drawLine(0,0,100,100); //绘画直线
}
常用的还有drawtext,drawrect

设置画笔
QPen pen(QColor(255,0,0);
painter.setPen(pen);
常用：setwidth(),PenStyle()

设置画刷
QBrush brush(QColor(0,255,0));
painter.setBrush(brush)

画图高级设置
painter.setRenderHint()里面可以选择很多属性，诸如抗锯齿之类的属性都在里面

改变画家的位置
painter.translate(100,0);移动画家位置
painter.save();保存画家位置
painter,restore()还原画家位置

绘画图片
painter.drawPixmap(posX,0,QPIxmap(":/image/xxx.png"));
如果要手动更新图片用update()

qt可以对画布进行更改
之前的绘画都是在widget上，因为widget继承了绘画设备。
绘画设备
Qpixmap 对图片显示做了优化
QPixmap map(300,300)
QPainter painter(&map);

Qimage 可以对图片进行像素级访问

QPicture 可以记录绘画的内容，保存成文件，用load可以加载
```

