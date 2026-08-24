# 15 Quy tắc trình bày báo cáo/luận văn (thầy Nhân góp ý — TT2 2026)

Checklist này đúc kết từ feedback của thầy Phan Trọng Nhân trên Báo cáo Thực tập 2.
Dùng lại khi viết **Đồ án tốt nghiệp (ĐATN)** và áp dụng cho cả **slide Beamer**.
Cột "Beamer" = có cần để ý khi làm slide không.

| # | Quy tắc | Cách làm đúng (LaTeX) | Beamer |
|---|---------|----------------------|--------|
| 1 | **Tóm tắt (Abstract)** phải có thứ tự: bài toán → phương pháp → kết quả đạt được ở giai đoạn này → *rồi mới* cấu trúc các chương. Không liệt kê chương ngay đầu. | Viết 3 đoạn ngắn (Bài toán / Phương pháp / Kết quả) trước phần "nội dung gồm N chương". | Slide mở đầu: 1 slide "Bài toán → Phương pháp → Kết quả" trước slide outline. |
| 2 | Mục lục phải có **Danh sách hình vẽ** và **Danh sách bảng** (hiển thị như mục "Tài liệu tham khảo"). | `\addcontentsline{toc}{chapter}{\listfigurename}` ngay trước `\listoffigures`; tương tự `\listtablename`. | N/A |
| 3 | Một mục **chỉ có 1 mục con thì bỏ mục con** đó (vd 3.1.4 chỉ có 3.1.4.1 → bỏ cấp con). | Chuyển `\subsubsection{X}` thành câu mở đầu in đậm `\textbf{X.}`. | Tránh slide chỉ có 1 bullet con. |
| 4 | Chương kết luận phải có **kế hoạch giai đoạn sau**: làm gì / bao lâu / như thế nào. Chỉ ghi việc **sẽ làm** (không liệt kê tất cả future work). | Bảng kế hoạch: cột GĐ / Nội dung / Phương pháp / Thời lượng. | Slide roadmap cuối: timeline ngắn gọn. |
| 5 | **Trích dẫn đầy đủ**, đặc biệt chương Cơ sở lý thuyết + Liên quan. Mọi chuẩn/dataset/mô hình nêu tên đều phải có `\cite` (vd "chuẩn RSNA 2024" → cite). | Thêm `~\cite{key}` ngay sau tên dataset/chuẩn/mô hình. | Footnote nguồn trên slide nếu nêu số liệu/chuẩn. |
| 6 | **Không để caption/title trùng** ở cả trong ảnh lẫn dưới hình. Nếu ảnh đã có title baked-in trùng caption → crop bỏ title trong ảnh. | Chỉ giữ `\caption{}` của LaTeX; export ảnh không kèm title (matplotlib: bỏ `plt.title`). | Bỏ title trong ảnh, dùng tiêu đề slide. |
| 7 | **Mọi Hình/Bảng phải được nhắc và mô tả trong text** (không để hình "mồ côi"). | Trong đoạn văn: "Hình~\ref{fig:x} minh họa…". Mỗi figure/table có ít nhất 1 câu `\ref`. | Mỗi hình trên slide nên có 1 câu giải thích. |
| 8 | **Trình bày nhất quán** khi tham chiếu Hình/Bảng (đừng chỗ in đậm chỗ không). Khuyến nghị: dùng `\ref` (KHÔNG hardcode số, KHÔNG bôi đậm). | Luôn `Hình~\ref{fig:x}` / `Bảng~\ref{tab:y}`. Tuyệt đối không gõ tay "**Hình 2.6**". Mỗi figure 1 `\label` **duy nhất**. | Dùng cách gọi nhất quán giữa các slide. |
| 9 | **Hình/Bảng lấy từ nguồn khác phải cite ở caption.** Hình tự tạo thì không cần. | `\caption{Mô tả \cite{nguon}.}` cho hình mượn. | "Nguồn: …" nhỏ dưới hình mượn. |
| 10 | **Chữ trong hình phải đủ to để đọc.** | Ưu tiên `width=\textwidth` (không dùng `scale=` nhỏ). Vẽ vector (TikZ/PDF) thay vì PNG mờ. Tăng fontsize khi export matplotlib. | Hình trên slide còn cần to hơn — chữ ≥ 16pt. |
| 11 | **Công thức nhất quán**: hoặc đánh số hết, hoặc không. Trong báo cáo khoa học → **đánh số tất cả**. | Dùng `\begin{equation}...\end{equation}` cho mọi công thức display. Tránh `$$...$$` (không số) xen kẽ. | Slide có thể không cần số, nhưng phải nhất quán. |
| 12 | **Title mục nên đầy đủ ý** (vd "Các mô hình tham khảo" → "**Ứng dụng** các mô hình tham khảo"). | Đặt tên section/subsection rõ hành động/mục đích. | Tiêu đề slide rõ nghĩa, không cụt. |
| 13 | **Gọi đúng tên tài liệu**: báo cáo thực tập thì xưng "báo cáo thực tập này", đừng gọi "luận văn này". | Tìm & sửa toàn bộ self-reference cho đúng loại tài liệu. | Footer slide ghi đúng "Báo cáo TT2" / "ĐATN". |
| 14 | **Hình kiến trúc/pipeline phải phản ánh đúng luồng đề xuất.** Nếu output A không phải input của bước B thì đừng vẽ như nối tiếp; nếu thiếu bước tích hợp thì phải thể hiện (kể cả là future work, vẽ nét đứt). Nên vẽ **toàn bộ giải pháp** (đạt mục tiêu chương 1) và đánh dấu phần đã làm / phần kế hoạch. | TikZ: nét liền = đã làm, nét đứt/xám = kế hoạch; chú thích trong caption. | Slide kiến trúc: highlight phần "đã làm" vs "sắp làm". |
| 15 | **Hạn chế trang nhiều dòng trắng.** | Bỏ `\newpage`/`\clearpage` ép trang không cần thiết. Float to dùng `[ht]`/`[htbp]` thay vì `[H]` cứng để LaTeX lấp đầy trang. Đặt hình/bảng gần chỗ tham chiếu. | Đừng để slide trống nửa dưới — chia nội dung đều. |

## Ghi nhớ nhanh (mnemonic)
- **Cấu trúc:** Tóm tắt theo (1), Mục lục đủ (2), bỏ mục con lẻ (3), kế hoạch rõ (4), title đủ ý (12), gọi đúng tên (13).
- **Hình/Bảng:** nhắc trong text (7), nhất quán `\ref` (8), cite nguồn (9), chữ to (10), không caption dư (6), kiến trúc đúng luồng (14).
- **Khác:** cite đầy đủ (5), công thức đánh số hết (11), không trang trắng (15).

## Snippet LaTeX hay dùng lại
```latex
% Rule 2 — đưa danh mục hình/bảng vào Mục lục
\tableofcontents
\clearpage
\addcontentsline{toc}{chapter}{\listfigurename}\listoffigures
\clearpage
\addcontentsline{toc}{chapter}{\listtablename}\listoftables

% Rule 8,9,10 — figure chuẩn
\begin{figure}[ht]            % Rule 15: [ht] thay vì [H]
    \centering
    \includegraphics[width=\textwidth]{hinh.png}  % Rule 10
    \caption{Mô tả hình \cite{nguon}.}            % Rule 9 (nếu mượn)
    \label{fig:ten_duy_nhat}                       % Rule 8: label duy nhất
\end{figure}
% Trong text: Hình~\ref{fig:ten_duy_nhat} cho thấy ...   % Rule 7,8

% Rule 11 — công thức luôn đánh số
\begin{equation}
    y = f(Wx + b)
\end{equation}
```
