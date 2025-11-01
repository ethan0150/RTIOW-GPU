This project is tested on both Windows 10 and 11 and should be able to run on any GPU that supports OpenGL 4.3. The following environment setup steps will be using Windows 11 as an example.
1. Download Visual Studio 2022 Community here: https://visualstudio.microsoft.com/zh-hant/downloads/
2. Run the installation file all the way while leaving all default options as is.
3. After the installation, find & open **Visual Studio Installer** in the start menu.
4. Click on **更多 > 匯入組態**, choose icg-final.vsconfig, click on **檢閱詳細資料** and click **修改**. Wait for it to download & install all the components before proceeding to the next step.
5. Open Powershell at C:\ and run `git clone https://github.com/microsoft/vcpkg.git`
6. Run `cd vcpkg; .\bootstrap-vcpkg.bat`
7. Run `.\vcpkg integrate install`
8. Run the following to install all the needed libraries
```cmd
.\vcpkg install glew
.\vcpkg install glfw3
.\vcpkg install glm
```
9. Open the RTIOW-GPU folder w/ Visual Studio and wait a few seconds for CMake generation to complete. There should be something like `CMake generation finished.` in the output at the bottom of the Visual Studio window on completion.
10. Click **Build > Build All**.
11. Navigate to RTIOW-GPU\out\build\x64-Debug\bin and open a terminal there.
12. Run `$executionTime = Measure-Command { .\raytracer.exe }`
13. output.ppm should now show up in the same directory. Run `$executionTime.TotalSeconds` to see how much time is consumed to generate it.
The below steps are for running the original CPU-based raytracer:
14. Clone this repo: https://github.com/RayTracing/raytracing.github.io.git
15. Open the repo w/ Visual Studio and wait a few seconds for CMake generation to complete. There should be something like `CMake generation finished.` in the output at the bottom of the Visual Studio window on completion.
16. Open the repo w/ Visaul Studio and click **Build > Build All**.
17. Navigate to raytracing.github.io\out\build\x64-Debug and open a terminal there.
18. Run `$executionTime = Measure-Command { .\inOneWeekend.exe > output.ppm }`
19. output.ppm should now show up in the same directory. Run `$executionTime.TotalSeconds` to see how much time is consumed to generate it.