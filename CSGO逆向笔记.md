# CS:GO

## 查找自己的矩阵

方法：不改变视角，进行移动操作后搜未变动的数值，改变视角并且不移动，操作其他东西后搜变动的数值

自己矩阵的特征：4x4、array[0][0]不开镜的时候小于1，第一次开镜大于1，第二次开镜大于等于5，自己的矩阵下面一行会跟着一行几乎一样的数据

实操：

1. 准备工作

   选择人机训练开始游戏，在CSGO控制台输入以下命令

   ```
   sv_cheats 1
   mp_startmoney 16000 出生金钱为16000
   mp_roundtime_defuse 60 休闲/竞技模式每局时间60分钟
   bot_stop 1  机器人暂停活动
   mp_restartgame 1 1秒后刷新游戏
   ```

   开局起一把大狙，ce附加到游戏进程
   ![image-20220813173407518](C:UsersROOTAppDataRoamingTyporatypora-user-imagesimage-20220813173407518.png)

根据上面的筛选条件，先筛选出大概的数据。随便点一个数据，选择查看周围的内存，此时改变视角，不是四乘四就过掉，

engine.dll+76 F90C

```
0.75 0.03 0.00 1340.29
-0.02 0.44 1.26 -261.23
-0.04 0.94 -0.33 123.38 
-0.04 0.94 -0.33 130.35
```

client.dll+4DC D234

```
0.75 0.03 0.00 1340.29
-0.02 0.44 1.26 -261.23 
-0.04 0.94 -0.33 123.38 
-0.04 0.94 -0.33 130.35
```

## 找自己的角度

方法：
角度的特征：

上下角度：在抬头到顶上的时候，角度为-89，在低头到最底下为89

左右角度：一般在上下角度的旁边，访问内存猜

实操：选择单浮点类型，抬头看天搜-89，低头看地搜89
<img src="C:UsersROOTAppDataRoamingTyporatypora-user-imagesimage-20220813180625331.png" alt="image-20220813180625331" style="zoom:50%;" />

把扫到的所有结果放在底下的地址栏，对半分修改数值，修改后如果视角变动则上下角度的值在其中，继续对半修改，直到找到正确数值
![image-20220813181144655](C:UsersROOTAppDataRoamingTyporatypora-user-imagesimage-20220813181144655.png)

根据上下角度，浏览相关内存区域，能找到左右角度

 ![image-20220813182023446](C:UsersROOTAppDataRoamingTyporatypora-user-imagesimage-20220813182023446.png)

将数据显示模式改为浮点型，可以在上下坐标(29.76)右侧，发现数据(17.87)，修改右侧数据，人物画面左右移动，则确认是左右角度
![image-20220813181925552](C:UsersROOTAppDataRoamingTyporatypora-user-imagesimage-20220813181925552.png)

故：左右角度 = 上下角度+4

### 寻找基址

对上下角度数据，找出是什么改写了这个地址

会出现两个，随便点一个显示反汇编，都能追到

```
engine.dll.text+21A01C - mov eax,[engine.dll.data+79FD8]
engine.dll.text+21A021 - movss [ebp-10],xmm0
engine.dll.text+21A026 - call dword ptr [eax+14]
engine.dll.text+21A029 - mov edx,[edi]
engine.dll.text+21A02B - mov esi,[eax*4+engine.dll.data+79FDC]
engine.dll.text+21A032 - lea ecx,[esi+00004D90]
engine.dll.text+21A038 - call engine.dll.text+2A6E70
engine.dll.text+21A03D - movss xmm0,[ebp-1C]
engine.dll.text+21A042 - lea ecx,[esi+00004D90]
engine.dll.text+21A048 - movss [esi+00004D90],xmm0
```

分析反汇编：

esi = FBF60010

上下坐标 = [esi+00004D90] = [[eax*4+engine.dll.data+79FDC]+00004D90]

eax 应该是 call dword ptr [eax+14] 的返回值，需要分析call，有些麻烦，后期写代码要分析call的传参，试试ce直接搜索esi，能不能得到基址

### ce找基址

选择四字节，十六进制，搜索FBF60010，直接搜到一个基址，直接用

 ![image-20220813184643757](C:UsersROOTAppDataRoamingTyporatypora-user-imagesimage-20220813184643757.png)

因此

上下角度 = [[engine.dll+58CFDC]+00004D90]
左右角度 = [[engine.dll+58CFDC]+00004D90]+4

 ![image-20220813185024366](C:UsersROOTAppDataRoamingTyporatypora-user-imagesimage-20220813185024366.png)

经验证，结果正确

## 找自己的坐标

方法1：模糊搜索，动一下搜一下

方法2：找一个高地，先模糊搜索出z轴的坐标。首先未知的初始值，上高地，z轴值增加了，下高地，z轴值减少了，找到一堆数据后，批量修改，找到正确的z轴值，根据正确的z轴值找到x轴，y轴，再找xyz的基址

实践：

随便找个坡，根据方法二找到一堆数据

 ![image-20220813185921237](C:UsersROOTAppDataRoamingTyporatypora-user-imagesimage-20220813185921237.png)

继续对半分，改数据，改到正确的数据人物会跳一下或者卡在空中，能找到三个数值
![image-20220813191422317](C:UsersROOTAppDataRoamingTyporatypora-user-imagesimage-20220813191422317.png)

根据修改的效果来看，第三个内存是更加自然的

### 找x轴和y轴

选择第一个地址，浏览周围的内存

![image-20220813191715606](C:UsersROOTAppDataRoamingTyporatypora-user-imagesimage-20220813191715606.png)

-11.45 前面就是 x轴和y轴，修改 1500，发现人物向前瞬移，是正确的人物坐标
因此

x坐标 = z坐标的地址-8

y坐标 = z坐标的地址-4

另外两个地址也能得到同样的效果

### 找基址

两个直接改变高度数值的地址，基址无法直接搜到，要分析汇编，懒得分析了，网上蹿一段距离可以用ce直接追到基址，网上蹿一段距离分析基址

选择网上蹿一段距离的地址，查看是什么改写了内存，游戏里跳一下，出现mov [edi+000001E4],eax  ，EDI=7808B580。

在ce里四字节，十六进制 搜索7808B580，得到两个基址，里面储存了7808B580，随便拿一个用，游戏重启后再验证一下。

 ![image-20220813193610686](C:UsersROOTAppDataRoamingTyporatypora-user-imagesimage-20220813193610686.png)

得到：

x轴 = [[server.dll+AC6258]+1E4-8]

y轴 = [[server.dll+AC6258]+1E4-4]

z轴 = [[server.dll+AC6258]+1E4]

 x = [client.dll+0x58CFDC]+3464

## 找敌人第坐标

方法：找自己个敌人的相对位置，当自己的高度80时，而敌人低于自己的时候ce搜索小于80的数值，敌人高于自己的时候搜大于80的数值，再批量修改数值，找到正确的数值，根据自己的z轴坐标，找到敌人的z轴坐标，再找到敌人坐标的x轴和y轴，

实践：

## 内存特征码找基址

找到数据的内存后，将内存显示方式改为字节，找上下不动的数据，作为特征码

```
#include <stdio.h>
#include<Windows.h>
#include<iostream>
#include <GLFW/glfw3.h>
#include"CSGOGO.h"
#include"offset.h"
#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"
#include "imgui/alifont.hpp"
using namespace hazedumper::netvars;
using namespace hazedumper::signatures;



//声明外部变量
extern DWORD g_process_id;
extern HANDLE g_process_handle;
extern UINT_PTR g_local_player;
extern UINT_PTR g_player_list_address;
extern UINT_PTR g_matrix_address;
extern UINT_PTR g_angle_address;
extern HWND g_game_hwnd;
extern module_information engine_module;
extern module_information client_module;
extern module_information server_module;

void DrawLine(Vec2 &start, Vec2 &end)
{
 glBegin(GL_LINES);
 //glColor4f(255, 0, 0, 255);
 glVertex2f(start.x, start.y);
 glVertex2f(end.x, end.y);
 glEnd();

}
void ShowMenu(GLFWwindow *Window)
{
 std::cout << "展示界面n";
 glfwSetWindowAttrib(Window, GLFW_MOUSE_PASSTHROUGH, GLFW_FALSE);
}
void HideMenu(GLFWwindow *Window)
{
 glfwSetWindowAttrib(Window, GLFW_MOUSE_PASSTHROUGH, GLFW_TRUE);
 std::cout << "隐藏界面n";


}

static void glfw_error_callback(int error, const char* description)
{
 fprintf(stderr, "Glfw Error %d: %sn", error, description);
}



int main(int, char**)
{
 /////////////////////////功能性代码////////////////////////////////////////////////////////////////////////////////////////////

 init_address("csgo.exe");
 player_list player_list64[64]{ 0 };
 get_player_list_info(player_list64);
 UINT_PTR temp_address;

 /////////////////////////功能性代码////////////////////////////////////////////////////////////////////////////////////////////



 // Setup window
 glfwSetErrorCallback(glfw_error_callback);
 if (!glfwInit())
  return 1;
 GLFWmonitor *monitor = glfwGetPrimaryMonitor();



 //###########################设置窗口###########################
 const char* glsl_version = "#version 130";
 int Height = glfwGetVideoMode(monitor)->height;
 int Width = glfwGetVideoMode(monitor)->width;
 glfwWindowHint(GLFW_FLOATING, true);
 glfwWindowHint(GLFW_RESIZABLE, false);
 glfwWindowHint(GLFW_MAXIMIZED, true);
 glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, true);
 //###########################设置窗口###########################




 GLFWwindow* window = glfwCreateWindow(Width, Height, "titile", NULL, NULL);
 if (window == NULL)
  return 1;
 ShowWindow(GetConsoleWindow(), SW_HIDE);
 glfwSetWindowAttrib(window, GLFW_DECORATED, false); //设置没有标题栏

 glfwMakeContextCurrent(window);
 glfwSwapInterval(1);
 IMGUI_CHECKVERSION();
 ImGui::CreateContext();
 ImGuiIO& io = ImGui::GetIO(); (void)io;
 ImFont* font = io.Fonts->AddFontFromMemoryTTF((void *)alifont_data, alifont_size, 18.0f, NULL, io.Fonts->GetGlyphRangesChineseFull());

 ImGui::StyleColorsDark();
 ImGui_ImplGlfw_InitForOpenGL(window, true);
 ImGui_ImplOpenGL3_Init(glsl_version);
 bool bMenuVisible = true;
 bool perspective = false;
 bool cheat = true;
 float temp_pos[3];
 float Matrix[16];
 Vec2 LineOrigin;
 LineOrigin.x = 0.0f;
 LineOrigin.y = -1.0f;
 while (!glfwWindowShouldClose(window))
 {
  glfwPollEvents();
  glClear(GL_COLOR_BUFFER_BIT);
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
  /****************************************界面代码*****************************************************************************/
  if (GetAsyncKeyState(VK_INSERT) & 1)
  {
   bMenuVisible = !bMenuVisible;
   if (bMenuVisible)
    ShowMenu(window);
   else
    HideMenu(window);
  }

  if (bMenuVisible)
  {
   ImGui::Text("Push 'Insert' to hide or show window");
   ImGui::Text(" ");
   ImGui::Checkbox("perspective", &perspective);
   ImGui::Text(" ");
   if (ImGui::Button("Pause"))
    cheat = false;
   ShowWindow(GetConsoleWindow(), SW_SHOW);

   ImGui::SameLine();
   if (ImGui::Button("Resume"))
   {
    ShowWindow(GetConsoleWindow(), SW_SHOW);
    init_address("csgo.exe");
    ShowWindow(GetConsoleWindow(), SW_HIDE);
    cheat = true;
   }
   ImGui::SameLine();

   if (ImGui::Button("exit"))
   {
    return 0;
   }
  }
/****************************************界面代码*****************************************************************************/
//以下为cheat代码
  if (cheat)
  {
   Vec2 ScreenCoord;
   ScreenCoord.x = 0.0f;
   ScreenCoord.y = -1.0f;
   ReadProcessMemory(g_process_handle, (LPCVOID)(client_module.module_address + dwViewMatrix), Matrix, sizeof(float) * 16, NULL);
   for (short int i = 0; i < 64; ++i)
   {
    ReadProcessMemory(g_process_handle, (LPCVOID)(g_local_player + m_iTeamNum), &temp_address, 4, NULL);
    int iTeamNum = temp_address;
    UINT_PTR Entity;
    bool bDormant;
    //获取敌人实体
    ReadProcessMemory(g_process_handle, (LPCVOID)(client_module.module_address + dwEntityList + i * 0x10), &temp_address, sizeof(float), NULL);
    Entity = temp_address;
    ReadProcessMemory(g_process_handle, (LPCVOID)(Entity + m_bDormant), &temp_address, sizeof(bool), NULL);
    bDormant = temp_address;
    ReadProcessMemory(g_process_handle, (LPCVOID)(Entity + m_iTeamNum), &temp_address, 4, NULL);
    int EntityTeamNum = temp_address;
    ReadProcessMemory(g_process_handle, (LPCVOID)(Entity + m_iHealth), &temp_address, 4, NULL);
    int blood = temp_address;
    if ((Entity == NULL) || (Entity == g_local_player) || bDormant || (EntityTeamNum == iTeamNum) || (blood <= 0))
     continue;
    //获取三维坐标
    ReadProcessMemory(g_process_handle, (LPVOID)(Entity + m_vecOrigin), &temp_pos, 12, NULL);
    Vec3 EntityLocation;
    EntityLocation.x = temp_pos[0];
    EntityLocation.y = temp_pos[1];
    EntityLocation.z = temp_pos[2];

    if (!WorldToScreen(EntityLocation, ScreenCoord, Matrix))
     continue;
    DrawLine(LineOrigin, ScreenCoord);
   }
  }
    // Rendering
  ImGui::Render();
  int display_w, display_h;
  glfwGetFramebufferSize(window, &display_w, &display_h);
  glViewport(0, 0, display_w, display_h);
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
  glfwSwapBuffers(window);
 }

 // Cleanup
 ImGui_ImplOpenGL3_Shutdown();
 ImGui_ImplGlfw_Shutdown();
 ImGui::DestroyContext();
 glfwDestroyWindow(window);
 glfwTerminate();

 return 0;
}

```

敌人骨骼：

[[client.dll+0x4DDD93C+0x10]+0x26A8]

![image-20221001235112815](C:UsersROOTAppDataRoamingTyporatypora-user-imagesimage-20221001235112815.png)

[[client.dll+0x4DDD93C+0x10]+0x26A8]+9C = 腰部x

[[client.dll+0x4DDD93C+0x10]+0x26A8]+9C+0x10 = 腰部y

[[client.dll+0x4DDD93C+0x10]+0x26A8]+9C+0x20 = 腰部z

+6C = 头部

+CC = 胸部

## 代码备份
```C++
#include <stdio.h>
#include<Windows.h>
#include<iostream>
#include <GLFW/glfw3.h>
#include"CSGOGO.h"
#include"offset.h"
#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"
#include<cmath>
#include<iomanip>
using namespace hazedumper::netvars;
using namespace hazedumper::signatures;
#define Head 0x6C
#define waist 0x9C
#define xiong 0xCC
#define xiong2 0xFC
#define bozi 0x12C
#define face 0x15C
#define etou 0x18C
#define bizi 0x1BC
#define bozi 0x1EC
#define 左肩膀 0x21C
#define 左胳膊肘 0x24C
#define 左手 0x27C
#define 左手腕 0x2AC
#define 左手背 0x2DC
#define 左手指 0x30C
#define 右手指 0x33C
#define 右手背 0x36C
#define 两手中心 0x39C
#define 两手下半部分 0x3CC
#define 右手下半部分 0x3FC
#define 左手腕2 0x42C
#define 右胳膊肘 0x7EC
#define 右手腕 0x81C
#define 左肩膀（靠脖子） 0x72C
#define 枪 0x99C
//声明外部变量
extern DWORD g_process_id;
extern HANDLE g_process_handle;
extern UINT_PTR g_local_player;
extern UINT_PTR g_player_list_address;
extern UINT_PTR g_matrix_address;
extern UINT_PTR g_angle_address;
extern HWND g_game_hwnd;
extern module_information engine_module;
extern module_information client_module;
extern module_information server_module;
extern float g_client_width;
extern float g_client_height;
void DrawLine(Vec2 &start, Vec2 &end)
{
 glBegin(GL_LINES);
 glColor4f(255, 255, 255, 100);
 glVertex2f(start.x, start.y);
 glVertex2f(end.x, end.y);
 glEnd();

}
void DrawAngle(Vec3 EntityHeadBone, Vec3 EntityLocation, Vec3 LeftBone, Vec3 RightBone, float Matrix[16])
{
 glBegin(GL_LINES);
 glColor4f(255, 0, 0, 255);
 Vec2 TD_EntityHeadBone, TD_EntityLocation, TD_LeftBone, TD_RightBone;
 WorldToScreen(EntityHeadBone, TD_EntityHeadBone, Matrix);
 WorldToScreen(EntityLocation, TD_EntityLocation, Matrix);
 WorldToScreen(LeftBone, TD_LeftBone, Matrix);
 WorldToScreen(RightBone, TD_RightBone, Matrix);
 //std::cout<< TD_EntityHeadBone.x<<","<< TD_EntityHeadBone.y << std::endl << TD_EntityLocation.x << "," << TD_EntityLocation.y << std::endl << TD_LeftBone.x << "," << TD_LeftBone.y << std::endl << TD_LeftBone.x << "," << TD_LeftBone.y<<std::endl<< TD_RightBone.x << "," << TD_RightBone.y;
 if ((TD_LeftBone.x >= 0 && TD_RightBone.x <= 0) || (TD_LeftBone.x <= 0 && TD_RightBone.x >= 0))
 {
  if (abs((int)TD_LeftBone.x) + abs((int)TD_RightBone.x) >= 0.35)
   return;
 }
 //else if(TD_RightBone.x<=0&& TD_LeftBone.x<=0)
 //{
 //  //std::cout << abs((int)TD_RightBone.x - (int)TD_LeftBone.x)<<std::endl;
 //}
 //else
 //{
 // if (abs(TD_LeftBone.x + TD_RightBone.x)>0.5)
 //  return;
 //}
 glVertex2f((TD_LeftBone.x), (TD_EntityHeadBone.y));
 glVertex2f(TD_LeftBone.x, TD_EntityLocation.y);
 
 glVertex2f(TD_LeftBone.x, TD_EntityHeadBone.y);
 glVertex2f(TD_RightBone.x, TD_EntityHeadBone.y);

 glVertex2f(TD_RightBone.x, TD_EntityHeadBone.y);
 glVertex2f(TD_RightBone.x, TD_EntityLocation.y);

 glVertex2f(TD_RightBone.x, TD_EntityLocation.y);
 glVertex2f(TD_LeftBone.x, TD_EntityLocation.y);

 glEnd();

}
void ShowMenu(GLFWwindow *Window)
{
// std::cout << "展示界面n";
 glfwSetWindowAttrib(Window, GLFW_MOUSE_PASSTHROUGH, GLFW_FALSE);
}
void HideMenu(GLFWwindow *Window)
{
 glfwSetWindowAttrib(Window, GLFW_MOUSE_PASSTHROUGH, GLFW_TRUE);
// std::cout << "隐藏界面n";
}
void Aimbot(Vec3 Pos,float Matrix[16])
{
 bool once = true;
 Vec2 ScreenPos;
 WorldToScreen(Pos, ScreenPos, Matrix);
 double distance = sqrt((ScreenPos.x)*(ScreenPos.x)+ (ScreenPos.y )*(ScreenPos.y));
 //std::cout << distance <<std::endl;
 std::cout <<std::setprecision(4)<< ScreenPos.x<<std::endl;
 /*
 Y轴相差的距离的关系
 差0.02
 则把角度-1
 差-0.02
 则把角度+1
 成
 差值  角度
 0.02   0
 0.00   -1
 -0.02  -2
 的关系

 x轴相差的距离关系
 当横向做视角坐标在-90~90之间
 差值每多0.02/0.01 则角度-1
 称
 0.09  27
 0.08  26
 0.06  25
 的趋势
 在
 */
 //画线测试 
 glBegin(GL_LINES);
 glVertex2f(0,0);
 glVertex2f(ScreenPos.x, ScreenPos.y);
 glEnd();
  if ((distance <= 1.0))
  {
   float a = 0;
   float angleX = 0;
   float angleY = 0;
   once = false;
   float currentAngleX, currentAngleY;
   
   if (GetAsyncKeyState(VK_F3)&1)
   {
    ReadProcessMemory(g_process_handle, (LPVOID)g_angle_address, &currentAngleX, 4, NULL);
    ReadProcessMemory(g_process_handle, (LPVOID)(g_angle_address + 4), &currentAngleY, 4, NULL);
    angleX = currentAngleY - (ScreenPos.x / 0.02);
    angleY = currentAngleX - (ScreenPos.y / 0.02)+1;
    WriteProcessMemory(g_process_handle, (LPVOID)g_angle_address, &angleY, 4, NULL);
    WriteProcessMemory(g_process_handle, (LPVOID)(g_angle_address + 4), &angleX, 4, NULL);
   }

  }
}
static void glfw_error_callback(int error, const char* description)
{
 fprintf(stderr, "Glfw Error %d: %sn", error, description);
}

void GetWindowSize()
{
 HDC hdc = GetDC(NULL);
  g_client_width = GetDeviceCaps(hdc, DESKTOPHORZRES);
  g_client_height = GetDeviceCaps(hdc, DESKTOPVERTRES);
 ReleaseDC(NULL, hdc);
}

void aimbo1t(float*EntityPos, Vec3 Pos, float Matrix[16])
{

 Vec2 ScreenPos;
 WorldToScreen(Pos, ScreenPos, Matrix);
 double distance = sqrt((ScreenPos.x)*(ScreenPos.x) + (ScreenPos.y)*(ScreenPos.y));
 float currentAngleX, currentAngleY;
 float temp_pos[3];
 float aimAngle[2];
 if (distance <= 0.5)
 {
  ReadProcessMemory(g_process_handle, (LPVOID)g_angle_address, &currentAngleX, 4, NULL);
  ReadProcessMemory(g_process_handle, (LPVOID)(g_angle_address + 4), &currentAngleY, 4, NULL);
  ReadProcessMemory(g_process_handle, (LPVOID)(g_local_player + m_vecOrigin), &temp_pos, 12, NULL);
  get_aimbot_angle(temp_pos, EntityPos, aimAngle);
  std::cout << aimAngle[0] << "," << aimAngle[1];

  if (abs((int)aimAngle[0] - (int)currentAngleX) > 30
   || abs((int)aimAngle[1] - (int)currentAngleY) > 30)
   return;

  WriteProcessMemory(g_process_handle, (LPVOID)g_angle_address, aimAngle, sizeof(float) * 2, NULL);

 }
 
}

int main(int, char**)
{

 /////////////////////////功能性代码////////////////////////////////////////////////////////////////////////////////////////////
 GetWindowSize();
 init_address("csgo.exe");
 UINT_PTR temp_address;

 /////////////////////////功能性代码////////////////////////////////////////////////////////////////////////////////////////////



 // Setup window
 glfwSetErrorCallback(glfw_error_callback);
 if (!glfwInit())
  return 1;
 GLFWmonitor *monitor = glfwGetPrimaryMonitor();



 //###########################设置窗口###########################
 const char* glsl_version = "#version 130";
 int Height = glfwGetVideoMode(monitor)->height;
 int Width = glfwGetVideoMode(monitor)->width;
 glfwWindowHint(GLFW_FLOATING, true);
 glfwWindowHint(GLFW_RESIZABLE, false);
 glfwWindowHint(GLFW_MAXIMIZED, true);
 glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, true);
 //###########################设置窗口###########################




 GLFWwindow* window = glfwCreateWindow(Width, Height, "titile", NULL, NULL);
 if (window == NULL)
  return 1;
 glfwSetWindowAttrib(window, GLFW_DECORATED, false); //设置没有标题栏
 //ShowWindow(GetConsoleWindow(), SW_HIDE);
 glfwMakeContextCurrent(window);
 glfwSwapInterval(1);
 IMGUI_CHECKVERSION();
 ImGui::CreateContext();
 ImGuiIO& io = ImGui::GetIO(); (void)io;
 ImGui::StyleColorsDark();
 ImGui_ImplGlfw_InitForOpenGL(window, true);
 ImGui_ImplOpenGL3_Init(glsl_version);
 bool bMenuVisible = true;
 bool Line = true;
 bool Angle = false;
 bool cheat = true;
 bool aimbot = false;
 bool bDomant;
 int lifestate;
 float temp_pos[3];
 float Matrix[16];
 float EntityPos[3];
 bool show = false;
 Vec2 LineOrigin;
 LineOrigin.x = 0.0f;
 LineOrigin.y = -1.0f;
 while (!glfwWindowShouldClose(window))
 {
  
  glfwPollEvents();
  glClear(GL_COLOR_BUFFER_BIT);
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
  /****************************************界面代码*****************************************************************************/
  if (GetAsyncKeyState(VK_F11) & 1)
  {
   bMenuVisible = !bMenuVisible;
   if (bMenuVisible)
    ShowMenu(window);
   else
    HideMenu(window);
  }
  if (GetAsyncKeyState(VK_F1) & 1)
  {
   Line = !Line;
  }
  if (GetAsyncKeyState(VK_F2) & 1)
  {
   Angle = !Angle;
  }
  if (bMenuVisible)
  {
   ImGui::Text("Push F11 To Hide Window");
   ImGui::Text("");
   ImGui::Checkbox("Line  (F1)", &Line);
   ImGui::Text("");
   ImGui::Checkbox("Angle  (F2)", &Angle);
   ImGui::Text("");
   ImGui::Checkbox("Aimbot  (F3)", &aimbot);
   if (ImGui::Button("Show Cmd"))
   {
    if (show)
    {
     ShowWindow(GetConsoleWindow(), SW_HIDE);
     show = false;
    }
    else
    {
     ShowWindow(GetConsoleWindow(), SW_SHOW);
     show = true;
    }
   }
   ImGui::SameLine();
   if(ImGui::Button("Hiden"))
   {
    bMenuVisible = !bMenuVisible;
    HideMenu(window);
   }
   ImGui::SameLine();
   if (ImGui::Button("exit"))  return 0;
  }
  if (cheat)
  {
   
   Vec2 ScreenCoord;
   ScreenCoord.x = 0.0f;
   ScreenCoord.y = -1.0f;
   ReadProcessMemory(g_process_handle, (LPVOID)(engine_module.module_address + dwClientState), &temp_address, 4, NULL);//[engine.dll + 58CFDC]+00004D90
   g_angle_address = temp_address + dwClientState_ViewAngles;
   ReadProcessMemory(g_process_handle, (LPCVOID)(client_module.module_address + dwViewMatrix), Matrix, sizeof(float) * 16, NULL);
   for (short int i = 0; i < 64; ++i)
   {
    ReadProcessMemory(g_process_handle, (LPVOID)(client_module.module_address + dwLocalPlayer), &temp_address, 4, NULL);
    g_local_player = temp_address;
    ReadProcessMemory(g_process_handle, (LPCVOID)(g_local_player + m_iTeamNum), &temp_address, 4, NULL);
    int iTeamNum = temp_address;
    UINT_PTR Entity;
    bool bDormant;
    //获取敌人实体
    ReadProcessMemory(g_process_handle, (LPCVOID)(client_module.module_address + dwEntityList + i * 0x10), &temp_address, sizeof(float), NULL);
    Entity = temp_address;
    ReadProcessMemory(g_process_handle, (LPCVOID)(Entity + bDormant), &bDomant, sizeof(bool), NULL);
    ReadProcessMemory(g_process_handle, (LPVOID)(Entity + m_lifeState), &lifestate, 4, NULL);
    ReadProcessMemory(g_process_handle, (LPCVOID)(Entity + m_iTeamNum), &temp_address, 4, NULL);
    int EntityTeamNum = temp_address;
    ReadProcessMemory(g_process_handle, (LPCVOID)(Entity + m_iHealth), &temp_address, 4, NULL);
    int blood = temp_address;

    if ((Entity == NULL) || (Entity == g_local_player)  ||(EntityTeamNum == iTeamNum)|| (blood <= 0)||lifestate||bDomant)
     continue;

    //获取三维坐标
    Vec3 EntityHeadBone;
    Vec3 EntityLocation;
    Vec3 LeftBone,RightBone;
    Vec3 FaceBone;
    
    GetBonePos(Entity, EntityHeadBone, Head);
    GetBonePos(Entity, LeftBone, 左胳膊肘);
    GetBonePos(Entity, RightBone, 右胳膊肘);
    GetBonePos(Entity, FaceBone, face);
    ReadProcessMemory(g_process_handle, (LPVOID)(Entity + m_vecOrigin), &temp_pos, 12, NULL);


    EntityLocation.x = temp_pos[0];
    EntityLocation.y = temp_pos[1];
    EntityLocation.z = temp_pos[2];

    //std::cout << EntityHeaderBone.x << std::endl << EntityHeaderBone.y << std::endl << EntityHeaderBone.z;;
    if (!WorldToScreen(EntityLocation, ScreenCoord, Matrix))
     continue;
    
    if (Line)
    {
     DrawLine(LineOrigin, ScreenCoord);
    }
    if (Angle)
    {
     DrawAngle(EntityHeadBone, EntityLocation, LeftBone, RightBone, Matrix);
    }
    if (show) //如果开启了控制台界面则输出信息
    {
     std::cout
      << "进程ID:" << g_process_id << std::endl
      << "进程句柄:" << g_process_handle << std::endl
      << "矩阵地址:" << g_matrix_address << std::endl
      << "本地人物地址:" << g_local_player << std::endl
      << "玩家列表地址:" << g_player_list_address << std::endl
      << "自己阵营:" << iTeamNum << std::endl
      << "此时遍历到的对象阵营:" << EntityTeamNum << std::endl
      << "此时遍历到的对象血量:" << blood;
    }
    if (aimbot)
    {
     int isShotFire, IsScoped;
     float F_EntityHeadBone[3];
     F_EntityHeadBone[0] = EntityHeadBone.x, F_EntityHeadBone[1] = EntityHeadBone.y, F_EntityHeadBone[2] = EntityHeadBone.z;
     ReadProcessMemory(g_process_handle, (LPVOID)(g_local_player + m_iShotsFired), &isShotFire, 4, NULL);
     ReadProcessMemory(g_process_handle, (LPVOID)(g_local_player + m_bIsScoped), &IsScoped, 4, NULL);
     if ((GetAsyncKeyState(VK_LBUTTON) & 0x8000) || isShotFire || IsScoped)
      aimbo1t(F_EntityHeadBone, EntityHeadBone, Matrix);//Aimbot(EntityHeadBone,Matrix);
    }

   }
  }
  /****************************************界面代码*****************************************************************************/
    // Rendering
  ImGui::Render();
  int display_w, display_h;
  glfwGetFramebufferSize(window, &display_w, &display_h);
  glViewport(0, 0, display_w, display_h);
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
  glfwSwapBuffers(window);
 }

 // Cleanup
 ImGui_ImplOpenGL3_Shutdown();
 ImGui_ImplGlfw_Shutdown();
 ImGui::DestroyContext();
 glfwDestroyWindow(window);
 glfwTerminate();

 return 0;
}

```
