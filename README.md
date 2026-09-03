# Diffusion Model Lab 01

Sinh ảnh bằng Stable Diffusion v1.5 và Hugging Face Diffusers trên GPU NVIDIA.

## Thiết lập bằng Python global 3.13

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

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
python generate_image.py --promt "..."
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
