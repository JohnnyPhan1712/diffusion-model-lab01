# Diffusion Model Lab 01

Sinh ảnh bằng Stable Diffusion v1.5 và Hugging Face Diffusers trên GPU NVIDIA.

## Yêu cầu cài đặt

Cấu hình đã được kiểm thử trong bài lab:

| Thành phần | Cấu hình |
|---|---|
| Hệ điều hành | Windows |
| Python | 3.13.15 (64-bit) |
| GPU | NVIDIA GeForce GTX 1650 Ti with Max-Q Design |
| VRAM | 4096 MiB (4 GB) |
| NVIDIA Driver | 595.79 |
| CUDA do driver hỗ trợ | 13.2 |
| CUDA của PyTorch | 12.8 |

Trước khi thiết lập dự án, kiểm tra Python và GPU:

```powershell
py --version
nvidia-smi
nvcc --version
```

Kết quả mong đợi trên máy đã kiểm thử:

```text
Python 3.13.15
NVIDIA GeForce GTX 1650 Ti with Max-Q Design
```

Không bắt buộc lệnh `nvcc --version` phải chạy thành công. `nvidia-smi` hiển thị
CUDA 13.2 là mức CUDA mà driver hỗ trợ, còn PyTorch wheel trong dự án đã kèm
CUDA runtime 12.8 cần thiết để sinh ảnh. Chỉ cần cài CUDA Toolkit riêng khi muốn
biên dịch PyTorch hoặc CUDA extension từ mã nguồn.

Nếu máy chưa có Python, tải Python 64-bit tại
[Python Releases for Windows](https://www.python.org/downloads/windows/). Khi cài,
nên bật Python Launcher để có thể sử dụng lệnh `py`.

## Thiết lập bằng Python global 3.13

Mở PowerShell tại thư mục dự án:

```powershell
cd D:\Workspace\Projects\diffusion-model-lab01
```

Tạo môi trường ảo độc lập từ Python global 3.13:

```powershell
py -3.13 -m venv .venv
```

Kích hoạt môi trường:

```powershell
.\.venv\Scripts\Activate.ps1
```

Nâng cấp `pip` và cài các thư viện đã khóa phiên bản trong `requirements.txt`:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Các thư viện chính được cài gồm:

- PyTorch 2.11.0 với CUDA 12.8.
- Torchvision 0.26.0.
- Diffusers 0.40.0.
- Transformers 5.16.1.
- Accelerate 1.14.0.
- Safetensors 0.8.0.

Kiểm tra dependency và khả năng truy cập GPU sau khi cài:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Kết quả kiểm thử thành công:

```text
No broken requirements found.
2.11.0+cu128
12.8
True
NVIDIA GeForce GTX 1650 Ti with Max-Q Design
```

Nếu PowerShell chặn `Activate.ps1`, không cần thay đổi Execution Policy; có thể
gọi trực tiếp `.\.venv\Scripts\python.exe` như các lệnh ở trên.

## Chạy chương trình đầy đủ

```powershell
.\.venv\Scripts\Activate.ps1
python generate_image.py
```

Mỗi lần chạy không truyền `--output`, chương trình tự lưu lần lượt thành
`outputs/output1.png`, `outputs/output2.png`, ... và không ghi đè ảnh cũ.

Muốn chỉ định tên cụ thể:

```powershell
python generate_image.py --output outputs\spiderman.png
```

Nhập prompt tạo ảnh theo ý muốn: 

```powershell
python generate_image.py --prompt "..."
```

## Chạy ví dụ đơn giản theo LAB01

```powershell
python generate_image_simple.py
```

File đơn giản dùng prompt `A futuristic city at night` và cũng tự đặt tên
`output1.png`, `output2.png`, ... Xem báo cáo chi tiết tại
`docs/LAB01_Diffusion_Model_Report.md`.

Model được lưu tại `.cache/huggingface/hub` trong chính dự án. Môi trường ảo dùng
Python global 3.13.15 và không phụ thuộc Python hoặc model cache của Codex.
