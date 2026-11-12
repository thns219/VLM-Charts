' ==============================================================================
' MACRO VẼ SƠ ĐỒ HỆ THỐNG ADAPTIVE-VLM-CHARTS CHUẨN 1:1 THEO HÌNH MẪU GỐC
' Không thay đổi cấu trúc, giữ nguyên 3 pipeline, các khối, mũi tên và ví dụ minh họa
' 
' CÁCH DÙNG TRONG POWERPOINT:
' 1. Mở PowerPoint -> Tạo Blank Presentation.
' 2. Nhấn Alt + F11 (mở cửa sổ VBA).
' 3. Chọn Insert -> Module.
' 4. Dán toàn bộ mã này vào và nhấn F5 (Run).
' ==============================================================================

Sub DrawOriginalAdaptiveVLMChartsDiagram()
    Dim pptPres As PowerPoint.Presentation
    Dim sld As PowerPoint.Slide
    Dim shp As PowerPoint.Shape
    Dim conn As PowerPoint.Shape
    
    Set pptPres = ActivePresentation
    ' Khổ chuẩn 16:9 (960 x 540 pt)
    pptPres.PageSetup.SlideWidth = 960
    pptPres.PageSetup.SlideHeight = 540
    
    Set sld = pptPres.Slides.Add(pptPres.Slides.Count + 1, ppLayoutBlank)
    sld.Background.Fill.Solid
    sld.Background.Fill.ForeColor.RGB = RGB(255, 255, 255)
    
    ' -------------------------------------------------------------
    ' 1. PANEL 1 (BÊN TRÁI): INPUT PIPELINE
    ' -------------------------------------------------------------
    ' Khung bao Panel 1
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 15, 15, 275, 415)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(248, 250, 252)
    shp.Line.ForeColor.RGB = RGB(203, 213, 225)
    shp.Line.Weight = 1.2
    
    ' Header Panel 1
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 25, 22, 255, 26)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(226, 232, 240)
    shp.Line.ForeColor.RGB = RGB(203, 213, 225)
    shp.TextFrame.TextRange.Text = "Adaptive-VLM-Charts: input pipeline"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Font.Size = 10.5
    shp.TextFrame.TextRange.Font.Bold = msoTrue
    shp.TextFrame.TextRange.Font.Color.RGB = RGB(30, 41, 59)
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    ' Khối: Bộ dữ liệu CQA (Scientific Chart / SLAKE)
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 35, 58, 235, 40)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(255, 255, 255)
    shp.Line.ForeColor.RGB = RGB(148, 163, 184)
    shp.TextFrame.TextRange.Text = "Bộ dữ liệu CQA" & vbCrLf & "(Scientific Chart / SLAKE)"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Lines(1).Font.Size = 9.5
    shp.TextFrame.TextRange.Lines(1).Font.Bold = msoTrue
    shp.TextFrame.TextRange.Lines(2).Font.Size = 8
    shp.TextFrame.TextRange.Lines(2).Font.Color.RGB = RGB(100, 116, 139)
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    ' Khối: Mẫu đầu vào (I, Q)
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 35, 116, 235, 62)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(255, 255, 255)
    shp.Line.ForeColor.RGB = RGB(148, 163, 184)
    shp.TextFrame.TextRange.Text = "Mẫu đầu vào (I, Q)"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Font.Size = 9
    shp.TextFrame.TextRange.Font.Bold = msoTrue
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    ' Ô con: Image I & Question Q
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 48, 140, 100, 30)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(241, 245, 249)
    shp.Line.ForeColor.RGB = RGB(148, 163, 184)
    shp.TextFrame.TextRange.Text = "Image I"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Font.Size = 8.5
    shp.TextFrame.TextRange.Font.Bold = msoTrue
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 158, 140, 100, 30)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(241, 245, 249)
    shp.Line.ForeColor.RGB = RGB(148, 163, 184)
    shp.TextFrame.TextRange.Text = "Question Q"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Font.Size = 8.5
    shp.TextFrame.TextRange.Font.Bold = msoTrue
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    ' Khối: Phân loại D1-D4, R1-R4
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 55, 192, 195, 28)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(255, 255, 255)
    shp.Line.ForeColor.RGB = RGB(148, 163, 184)
    shp.TextFrame.TextRange.Text = "Phân loại      D1–D4       R1–R4"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Font.Size = 8.5
    shp.TextFrame.TextRange.Font.Bold = msoTrue
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    ' Khối biểu đồ mini
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 35, 232, 235, 185)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(255, 255, 255)
    shp.Line.ForeColor.RGB = RGB(203, 213, 225)
    shp.TextFrame.TextRange.Text = "■ xanh (Right)   □ vàng (Centre)"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Font.Size = 8.5
    shp.TextFrame.TextRange.Font.Bold = msoTrue
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    ' Ô câu hỏi trong biểu đồ
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 45, 345, 215, 60)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(254, 243, 199)
    shp.Line.ForeColor.RGB = RGB(245, 158, 11)
    shp.TextFrame.TextRange.Text = "Question: Hiệu giữa tổng các cột xanh và tổng các cột vàng là bao nhiêu?"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Font.Size = 8
    shp.TextFrame.TextRange.Font.Bold = msoTrue
    shp.TextFrame.TextRange.Font.Color.RGB = RGB(146, 64, 14)
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    ' -------------------------------------------------------------
    ' 2. PANEL 2 (Ở GIỮA): RETRIEVAL & REASONING PIPELINE
    ' -------------------------------------------------------------
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 305, 15, 385, 415)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(248, 250, 252)
    shp.Line.ForeColor.RGB = RGB(203, 213, 225)
    shp.Line.Weight = 1.2
    
    ' Header Panel 2
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 315, 22, 365, 26)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(226, 232, 240)
    shp.Line.ForeColor.RGB = RGB(203, 213, 225)
    shp.TextFrame.TextRange.Text = "Adaptive-VLM-Charts: retrieval & reasoning pipeline"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Font.Size = 10.5
    shp.TextFrame.TextRange.Font.Bold = msoTrue
    shp.TextFrame.TextRange.Font.Color.RGB = RGB(30, 41, 59)
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    ' Khối: Input (I, Q)
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 325, 60, 95, 40)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(255, 255, 255)
    shp.Line.ForeColor.RGB = RGB(148, 163, 184)
    shp.TextFrame.TextRange.Text = "Input" & vbCrLf & "(I, Q)"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Font.Size = 9
    shp.TextFrame.TextRange.Font.Bold = msoTrue
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    ' Khung nét đứt: Hybrid RAG
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 445, 55, 235, 115)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(250, 245, 255)
    shp.Line.ForeColor.RGB = RGB(124, 58, 237)
    shp.Line.DashStyle = msoLineDash
    shp.TextFrame.TextRange.Text = "Hybrid RAG"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Font.Size = 9.5
    shp.TextFrame.TextRange.Font.Bold = msoTrue
    shp.TextFrame.TextRange.Font.Color.RGB = RGB(124, 58, 237)
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignLeft
    
    ' Con trong Hybrid RAG: ChromaDB, CLIP, BM25
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 460, 78, 205, 30)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(255, 255, 255)
    shp.Line.ForeColor.RGB = RGB(192, 132, 252)
    shp.TextFrame.TextRange.Text = "ChromaDB"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Font.Size = 8.5
    shp.TextFrame.TextRange.Font.Bold = msoTrue
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 460, 115, 100, 45)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(255, 255, 255)
    shp.Line.ForeColor.RGB = RGB(192, 132, 252)
    shp.TextFrame.TextRange.Text = "CLIP ViT-L/14" & vbCrLf & "(Letterbox)"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Font.Size = 8
    shp.TextFrame.TextRange.Font.Bold = msoTrue
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 565, 115, 100, 45)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(255, 255, 255)
    shp.Line.ForeColor.RGB = RGB(192, 132, 252)
    shp.TextFrame.TextRange.Text = "BM25" & vbCrLf & "Caption"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Font.Size = 8
    shp.TextFrame.TextRange.Font.Bold = msoTrue
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    ' Ô Prompt D
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 520, 185, 85, 28)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(254, 249, 195)
    shp.Line.ForeColor.RGB = RGB(234, 179, 8)
    shp.TextFrame.TextRange.Text = "Prompt D"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Font.Size = 8.5
    shp.TextFrame.TextRange.Font.Bold = msoTrue
    shp.TextFrame.TextRange.Font.Color.RGB = RGB(133, 77, 14)
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    ' Đường Oracle / Direct Bypass
    Set shp = sld.Shapes.AddShape(msoShapeRectangle, 325, 145, 110, 20)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(240, 253, 250)
    shp.Line.ForeColor.RGB = RGB(13, 148, 136)
    shp.TextFrame.TextRange.Text = "Oracle / Direct Bypass"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Font.Size = 7.5
    shp.TextFrame.TextRange.Font.Bold = msoTrue
    shp.TextFrame.TextRange.Font.Color.RGB = RGB(13, 148, 136)
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    ' Khối VLM Qwen2.5-VL-7B
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 325, 230, 345, 185)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(255, 255, 255)
    shp.Line.ForeColor.RGB = RGB(148, 163, 184)
    shp.TextFrame.TextRange.Text = "VLM" & vbCrLf & "Qwen2.5-VL-7B"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Lines(1).Font.Size = 10.5
    shp.TextFrame.TextRange.Lines(1).Font.Bold = msoTrue
    shp.TextFrame.TextRange.Lines(2).Font.Size = 8.5
    shp.TextFrame.TextRange.Lines(2).Font.Color.RGB = RGB(100, 116, 139)
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    ' 3 chế độ con
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 345, 280, 305, 30)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(248, 250, 252)
    shp.Line.ForeColor.RGB = RGB(203, 213, 225)
    shp.TextFrame.TextRange.Text = "(a) Direct Zero-Shot"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Font.Size = 8.5
    shp.TextFrame.TextRange.Font.Bold = msoTrue
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 345, 320, 305, 30)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(248, 250, 252)
    shp.Line.ForeColor.RGB = RGB(203, 213, 225)
    shp.TextFrame.TextRange.Text = "(b) Standard CoT (2-stage)"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Font.Size = 8.5
    shp.TextFrame.TextRange.Font.Bold = msoTrue
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 345, 360, 305, 38)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(255, 247, 237)
    shp.Line.ForeColor.RGB = RGB(234, 88, 12)
    shp.Line.Weight = 1.4
    shp.TextFrame.TextRange.Text = "(c) Adaptive LoRA CoT (Đề xuất)"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Font.Size = 9
    shp.TextFrame.TextRange.Font.Bold = msoTrue
    shp.TextFrame.TextRange.Font.Color.RGB = RGB(234, 88, 12)
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    ' -------------------------------------------------------------
    ' 3. PANEL 3 (BÊN PHẢI): EVALUATION PIPELINE
    ' -------------------------------------------------------------
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 705, 15, 240, 415)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(248, 250, 252)
    shp.Line.ForeColor.RGB = RGB(203, 213, 225)
    shp.Line.Weight = 1.2
    
    ' Header Panel 3
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 715, 22, 220, 26)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(226, 232, 240)
    shp.Line.ForeColor.RGB = RGB(203, 213, 225)
    shp.TextFrame.TextRange.Text = "Adaptive-VLM-Charts: evaluation pipeline"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Font.Size = 10.5
    shp.TextFrame.TextRange.Font.Bold = msoTrue
    shp.TextFrame.TextRange.Font.Color.RGB = RGB(30, 41, 59)
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    ' Rationale R & Answer A
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 735, 60, 180, 28)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(255, 255, 255)
    shp.Line.ForeColor.RGB = RGB(148, 163, 184)
    shp.TextFrame.TextRange.Text = "Rationale R"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Font.Size = 8.5
    shp.TextFrame.TextRange.Font.Bold = msoTrue
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 735, 100, 180, 32)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(255, 255, 255)
    shp.Line.ForeColor.RGB = RGB(148, 163, 184)
    shp.TextFrame.TextRange.Text = "Answer A"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Font.Size = 9
    shp.TextFrame.TextRange.Font.Bold = msoTrue
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    ' Khối 1: Chuẩn hóa (Normalization)
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 735, 140, 180, 60)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(255, 255, 255)
    shp.Line.ForeColor.RGB = RGB(148, 163, 184)
    shp.TextFrame.TextRange.Text = "Chuẩn hóa (Normalization)" & vbCrLf & _
                                   "• Word-to-Number   • Bỏ LaTeX" & vbCrLf & _
                                   "• Medical Semantic Normalization"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Lines(1).Font.Size = 9
    shp.TextFrame.TextRange.Lines(1).Font.Bold = msoTrue
    shp.TextFrame.TextRange.Lines(2).Font.Size = 8
    shp.TextFrame.TextRange.Lines(3).Font.Size = 7.5
    shp.TextFrame.TextRange.Lines(3).Font.Color.RGB = RGB(100, 116, 139)
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignLeft
    
    ' Khối 2: Đánh giá CharXiv (Khớp mẫu & Số học)
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 735, 208, 180, 105)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(255, 255, 255)
    shp.Line.ForeColor.RGB = RGB(37, 99, 235)
    shp.Line.Weight = 1.2
    shp.TextFrame.TextRange.Text = "Đánh giá CharXiv (Khớp mẫu & Số học)" & vbCrLf & _
                                   "match_fixed (±5%)  |  Bootstrap 95% CI" & vbCrLf & _
                                   "Phân rã theo 4 nhóm câu hỏi:" & vbCrLf & _
                                   "• Number-in-Chart   • Number-in-Gen" & vbCrLf & _
                                   "• Text-in-Chart       • Text-in-Gen"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Lines(1).Font.Size = 8.5
    shp.TextFrame.TextRange.Lines(1).Font.Bold = msoTrue
    shp.TextFrame.TextRange.Lines(1).Font.Color.RGB = RGB(29, 78, 216)
    shp.TextFrame.TextRange.Lines(2).Font.Size = 7.5
    shp.TextFrame.TextRange.Lines(2).Font.Bold = msoTrue
    shp.TextFrame.TextRange.Lines(2).Font.Color.RGB = RGB(30, 64, 175)
    shp.TextFrame.TextRange.Lines(3).Font.Size = 7.5
    shp.TextFrame.TextRange.Lines(3).Font.Bold = msoTrue
    shp.TextFrame.TextRange.Lines(4).Font.Size = 7.5
    shp.TextFrame.TextRange.Lines(5).Font.Size = 7.5
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignLeft
    
    ' Khối 3: Đánh giá SLAKE (Y khoa & Sinh NLG)
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 735, 321, 180, 94)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(240, 253, 244)
    shp.Line.ForeColor.RGB = RGB(22, 163, 74)
    shp.Line.Weight = 1.2
    shp.TextFrame.TextRange.Text = "Đánh giá SLAKE (Y khoa & Sinh NLG)" & vbCrLf & _
                                   "• Closed: Strict Accuracy & Medical Acc" & vbCrLf & _
                                   "• Open NLG: BLEU-1, 2  |  ROUGE-1, L" & vbCrLf & _
                                   "• Strict EM & Medical Semantic Acc"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Lines(1).Font.Size = 8.5
    shp.TextFrame.TextRange.Lines(1).Font.Bold = msoTrue
    shp.TextFrame.TextRange.Lines(1).Font.Color.RGB = RGB(21, 128, 61)
    shp.TextFrame.TextRange.Lines(2).Font.Size = 7.5
    shp.TextFrame.TextRange.Lines(3).Font.Size = 7.8
    shp.TextFrame.TextRange.Lines(3).Font.Bold = msoTrue
    shp.TextFrame.TextRange.Lines(3).Font.Color.RGB = RGB(21, 128, 61)
    shp.TextFrame.TextRange.Lines(4).Font.Size = 7.2
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignLeft
    
    ' -------------------------------------------------------------
    ' 4. HỘP VÍ DỤ MINH HỌA PHÍA DƯỚI (BOTTOM EXEMPLAR)
    ' -------------------------------------------------------------
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 15, 440, 930, 48)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(254, 252, 232)
    shp.Line.ForeColor.RGB = RGB(234, 179, 8)
    shp.Line.Weight = 1.2
    shp.TextFrame.TextRange.Text = "Ví dụ Adaptive LoRA CoT: Step 1 (Target): cột xanh = nhãn ""Right"", cột vàng = nhãn ""Centre"". " & _
                                   "Step 2 (Grounding): xanh = 32 + 38 + 30 = 100; vàng = 55 + 27 + 18 = 100. " & _
                                   "Step 3 (Deduction): 100 − 100 = 0.  Answer: 0"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Font.Size = 8.5
    shp.TextFrame.TextRange.Font.Bold = msoTrue
    shp.TextFrame.TextRange.Font.Color.RGB = RGB(133, 77, 14)
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    ' -------------------------------------------------------------
    ' 5. CAPTION BÀI BÁO TẠP CHÍ
    ' -------------------------------------------------------------
    Set shp = sld.Shapes.AddTextbox(msoTextOrientationHorizontal, 15, 495, 930, 30)
    shp.TextFrame.TextRange.Text = "Hình 1. Tổng quan hệ thống Adaptive-VLM-Charts. Pipeline đầu vào tạo mẫu (I, Q); pipeline truy hồi & lập luận kết hợp Hybrid RAG " & _
                                   "với ba chế độ suy luận của VLM (đường Oracle đi thẳng vào prompt); pipeline đánh giá chuẩn hóa đáp án, chấm điểm match_fixed (CharXiv) " & _
                                   "và đánh giá độ đo sinh ngôn ngữ BLEU-1, BLEU-2, ROUGE-1, ROUGE-L (SLAKE Med-VQA)."
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Font.Size = 7.8
    shp.TextFrame.TextRange.Font.Italic = msoTrue
    shp.TextFrame.TextRange.Font.Color.RGB = RGB(71, 85, 105)
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignLeft
    
    MsgBox "Đã vẽ thành công sơ đồ hệ thống theo đúng cấu trúc hình mẫu!", vbInformation, "Thành Công"
End Sub
