' ==============================================================================
' MACRO VẼ SƠ ĐỒ HỆ THỐNG ADAPTIVE-VLM-CHARTS CHUẨN 1:1 THEO THIẾT KẾ MỚI NHẤT
' Tự động tạo Native Shapes trong PowerPoint, cho phép chỉnh sửa từng chữ, từng hộp, từng mũi tên.
'
' CÁCH DÙNG TRONG POWERPOINT:
' 1. Mở PowerPoint -> Tạo Blank Presentation.
' 2. Nhấn Alt + F11 (mở cửa sổ VBA).
' 3. Chọn Insert -> Module.
' 4. Dán toàn bộ mã này vào và nhấn F5 (Run).
' ==============================================================================

Sub DrawExactAdaptiveVLMChartsArchitecture()
    Dim pptPres As PowerPoint.Presentation
    Dim sld As PowerPoint.Slide
    Dim shp As PowerPoint.Shape
    
    Set pptPres = ActivePresentation
    ' Khổ chuẩn 16:9 (960 x 540 pt)
    pptPres.PageSetup.SlideWidth = 960
    pptPres.PageSetup.SlideHeight = 540
    
    Set sld = pptPres.Slides.Add(pptPres.Slides.Count + 1, ppLayoutBlank)
    sld.Background.Fill.Solid
    sld.Background.Fill.ForeColor.RGB = RGB(255, 255, 255)
    
    ' ==========================================================================
    ' 1. PANEL 1 (BÊN TRÁI): INPUT PIPELINE
    ' ==========================================================================
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 15, 12, 260, 480)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(255, 255, 255)
    shp.Line.ForeColor.RGB = RGB(15, 23, 42)
    shp.Line.Weight = 1.3
    
    ' Header Panel 1
    Set shp = sld.Shapes.AddShape(msoShapeRectangle, 20, 16, 250, 22)
    shp.Fill.Transparency = 1
    shp.Line.Visible = msoFalse
    shp.TextFrame.TextRange.Text = "Adaptive-VLM-Charts: input pipeline"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Font.Size = 10
    shp.TextFrame.TextRange.Font.Bold = msoTrue
    shp.TextFrame.TextRange.Font.Color.RGB = RGB(15, 23, 42)
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    ' Khối: Bộ dữ liệu (CharXiv / SLAKE)
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 28, 42, 234, 40)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(224, 242, 254)
    shp.Line.ForeColor.RGB = RGB(56, 189, 248)
    shp.TextFrame.TextRange.Text = "Bộ dữ liệu" & vbCrLf & "(CharXiv / SLAKE)"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Lines(1).Font.Size = 9.5
    shp.TextFrame.TextRange.Lines(1).Font.Bold = msoTrue
    shp.TextFrame.TextRange.Lines(2).Font.Size = 8
    shp.TextFrame.TextRange.Lines(2).Font.Color.RGB = RGB(51, 65, 85)
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    ' Mũi tên xuống Mẫu đầu vào
    Set shp = sld.Shapes.AddConnector(msoConnectorStraight, 145, 82, 145, 96)
    shp.Line.ForeColor.RGB = RGB(15, 23, 42)
    shp.Line.EndArrowheadStyle = msoArrowheadTriangle
    
    ' Khối: Mẫu đầu vào: (I, Q), I hoặc Q
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 28, 98, 234, 116)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(255, 255, 255)
    shp.Line.ForeColor.RGB = RGB(30, 41, 59)
    shp.TextFrame.TextRange.Text = "Mẫu đầu vào: (I, Q), I hoặc Q"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Font.Size = 8.8
    shp.TextFrame.TextRange.Font.Bold = msoTrue
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    shp.TextFrame.VerticalAnchor = msoAnchorTop
    
    ' Ô con: Image I (tùy chọn) - Nét đứt
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 38, 124, 94, 38)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(255, 255, 255)
    shp.Line.ForeColor.RGB = RGB(71, 85, 105)
    shp.Line.DashStyle = msoLineDash
    shp.TextFrame.TextRange.Text = "Image I" & vbCrLf & "(tùy chọn)"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Lines(1).Font.Size = 8.5
    shp.TextFrame.TextRange.Lines(1).Font.Bold = msoTrue
    shp.TextFrame.TextRange.Lines(2).Font.Size = 7.2
    shp.TextFrame.TextRange.Lines(2).Font.Italic = msoTrue
    shp.TextFrame.TextRange.Lines(2).Font.Color.RGB = RGB(100, 116, 139)
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    ' Ô con: Question Q (tùy chọn) - Nét đứt
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 38, 168, 94, 38)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(255, 255, 255)
    shp.Line.ForeColor.RGB = RGB(71, 85, 105)
    shp.Line.DashStyle = msoLineDash
    shp.TextFrame.TextRange.Text = "Question Q" & vbCrLf & "(tùy chọn)"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Lines(1).Font.Size = 8.5
    shp.TextFrame.TextRange.Lines(1).Font.Bold = msoTrue
    shp.TextFrame.TextRange.Lines(2).Font.Size = 7.2
    shp.TextFrame.TextRange.Lines(2).Font.Italic = msoTrue
    shp.TextFrame.TextRange.Lines(2).Font.Color.RGB = RGB(100, 116, 139)
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    ' Ô con: Phân loại D1-D4 / R1-R4
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 150, 124, 100, 82)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(255, 255, 255)
    shp.Line.ForeColor.RGB = RGB(30, 41, 59)
    shp.TextFrame.TextRange.Text = "Phân loại" & vbCrLf & vbCrLf & "D1–D4" & vbCrLf & "R1–R4"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Lines(1).Font.Size = 8.5
    shp.TextFrame.TextRange.Lines(1).Font.Bold = msoTrue
    shp.TextFrame.TextRange.Lines(3).Font.Size = 8
    shp.TextFrame.TextRange.Lines(3).Font.Bold = msoTrue
    shp.TextFrame.TextRange.Lines(4).Font.Size = 8
    shp.TextFrame.TextRange.Lines(4).Font.Bold = msoTrue
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    ' Khung nét đứt cam/vàng chứa Biểu đồ
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 28, 226, 234, 254)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(255, 255, 255)
    shp.Line.ForeColor.RGB = RGB(245, 158, 11)
    shp.Line.DashStyle = msoLineDash
    shp.Line.Weight = 1.2
    
    ' Biểu đồ Bar chart mini
    ' Cột 1
    Set shp = sld.Shapes.AddShape(msoShapeRectangle, 65, 375, 18, 50)
    shp.Fill.ForeColor.RGB = RGB(37, 99, 235)
    shp.Line.Visible = msoFalse
    Set shp = sld.Shapes.AddShape(msoShapeRectangle, 85, 395, 18, 30)
    shp.Fill.ForeColor.RGB = RGB(251, 191, 36)
    shp.Line.ForeColor.RGB = RGB(217, 119, 6)
    
    ' Cột 2
    Set shp = sld.Shapes.AddShape(msoShapeRectangle, 125, 345, 18, 80)
    shp.Fill.ForeColor.RGB = RGB(37, 99, 235)
    shp.Line.Visible = msoFalse
    Set shp = sld.Shapes.AddShape(msoShapeRectangle, 145, 360, 18, 65)
    shp.Fill.ForeColor.RGB = RGB(251, 191, 36)
    shp.Line.ForeColor.RGB = RGB(217, 119, 6)
    
    ' Cột 3
    Set shp = sld.Shapes.AddShape(msoShapeRectangle, 185, 385, 18, 40)
    shp.Fill.ForeColor.RGB = RGB(37, 99, 235)
    shp.Line.Visible = msoFalse
    Set shp = sld.Shapes.AddShape(msoShapeRectangle, 205, 400, 18, 25)
    shp.Fill.ForeColor.RGB = RGB(251, 191, 36)
    shp.Line.ForeColor.RGB = RGB(217, 119, 6)
    
    ' Trục x
    Set shp = sld.Shapes.AddLine(50, 425, 238, 425)
    shp.Line.ForeColor.RGB = RGB(148, 163, 184)
    shp.Line.Weight = 1.1
    
    ' Câu hỏi dưới biểu đồ
    Set shp = sld.Shapes.AddShape(msoShapeRectangle, 32, 435, 226, 38)
    shp.Fill.Transparency = 1
    shp.Line.Visible = msoFalse
    shp.TextFrame.TextRange.Text = "Question: Hiệu giữa tổng các cột xanh và tổng các cột vàng là bao nhiêu?"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Font.Size = 7.5
    shp.TextFrame.TextRange.Font.Bold = msoTrue
    shp.TextFrame.TextRange.Font.Color.RGB = RGB(15, 23, 42)
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    ' ==========================================================================
    ' 2. PANEL 2 (TRÊN PHẢI): RETRIEVAL, ROUTING & REASONING PIPELINE
    ' ==========================================================================
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 290, 12, 655, 240)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(255, 255, 255)
    shp.Line.ForeColor.RGB = RGB(15, 23, 42)
    shp.Line.Weight = 1.3
    
    ' Header Panel 2
    Set shp = sld.Shapes.AddShape(msoShapeRectangle, 310, 16, 615, 22)
    shp.Fill.Transparency = 1
    shp.Line.Visible = msoFalse
    shp.TextFrame.TextRange.Text = "Adaptive-VLM-Charts: retrieval, routing & reasoning pipeline"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Font.Size = 10
    shp.TextFrame.TextRange.Font.Bold = msoTrue
    shp.TextFrame.TextRange.Font.Color.RGB = RGB(15, 23, 42)
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    ' Mũi tên từ Panel 1 sang Panel 2
    Set shp = sld.Shapes.AddConnector(msoConnectorStraight, 262, 156, 305, 96)
    shp.Line.ForeColor.RGB = RGB(15, 23, 42)
    shp.Line.EndArrowheadStyle = msoArrowheadTriangle
    
    ' Khối: Input I và/hoặc Q
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 305, 72, 58, 48)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(255, 255, 255)
    shp.Line.ForeColor.RGB = RGB(30, 41, 59)
    shp.TextFrame.TextRange.Text = "Input" & vbCrLf & "I và/hoặc Q"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Lines(1).Font.Size = 8.5
    shp.TextFrame.TextRange.Lines(1).Font.Bold = msoTrue
    shp.TextFrame.TextRange.Lines(2).Font.Size = 7.2
    shp.TextFrame.TextRange.Lines(2).Font.Color.RGB = RGB(51, 65, 85)
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    ' Đường Oracle Bypass (nét đứt)
    Set shp = sld.Shapes.AddConnector(msoConnectorStraight, 334, 72, 334, 46)
    shp.Line.ForeColor.RGB = RGB(51, 65, 85)
    shp.Line.DashStyle = msoLineDash
    Set shp = sld.Shapes.AddConnector(msoConnectorStraight, 334, 46, 545, 46)
    shp.Line.ForeColor.RGB = RGB(51, 65, 85)
    shp.Line.DashStyle = msoLineDash
    Set shp = sld.Shapes.AddConnector(msoConnectorStraight, 545, 46, 545, 76)
    shp.Line.ForeColor.RGB = RGB(51, 65, 85)
    shp.Line.DashStyle = msoLineDash
    shp.Line.EndArrowheadStyle = msoArrowheadTriangle
    
    Set shp = sld.Shapes.AddShape(msoShapeRectangle, 345, 30, 190, 16)
    shp.Fill.Transparency = 1
    shp.Line.Visible = msoFalse
    shp.TextFrame.TextRange.Text = "Oracle (cận trên) / Direct Bypass"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Font.Size = 7.5
    shp.TextFrame.TextRange.Font.Italic = msoTrue
    shp.TextFrame.TextRange.Font.Bold = msoTrue
    shp.TextFrame.TextRange.Font.Color.RGB = RGB(30, 41, 59)
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    ' Mũi tên từ Input vào Hybrid RAG
    Set shp = sld.Shapes.AddConnector(msoConnectorStraight, 363, 96, 385, 96)
    shp.Line.ForeColor.RGB = RGB(15, 23, 42)
    shp.Line.EndArrowheadStyle = msoArrowheadTriangle
    
    ' Khối: Hybrid RAG
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 385, 46, 115, 114)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(255, 255, 255)
    shp.Line.ForeColor.RGB = RGB(30, 41, 59)
    shp.TextFrame.TextRange.Text = "Hybrid RAG" & vbCrLf & _
                                   "ChromaDB" & vbCrLf & _
                                   "CLIP ViT-L/14" & vbCrLf & _
                                   "(Letterbox)" & vbCrLf & _
                                   "BM25 Caption" & vbCrLf & _
                                   "(Query: text Q/I)"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Lines(1).Font.Size = 8.5
    shp.TextFrame.TextRange.Lines(1).Font.Bold = msoTrue
    shp.TextFrame.TextRange.Lines(2).Font.Size = 7.5
    shp.TextFrame.TextRange.Lines(3).Font.Size = 7.5
    shp.TextFrame.TextRange.Lines(4).Font.Size = 7
    shp.TextFrame.TextRange.Lines(4).Font.Color.RGB = RGB(100, 116, 139)
    shp.TextFrame.TextRange.Lines(5).Font.Size = 7.5
    shp.TextFrame.TextRange.Lines(6).Font.Size = 6.8
    shp.TextFrame.TextRange.Lines(6).Font.Italic = msoTrue
    shp.TextFrame.TextRange.Lines(6).Font.Color.RGB = RGB(100, 116, 139)
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    ' Mũi tên từ RAG vào Prompt D
    Set shp = sld.Shapes.AddConnector(msoConnectorStraight, 500, 96, 520, 96)
    shp.Line.ForeColor.RGB = RGB(15, 23, 42)
    shp.Line.EndArrowheadStyle = msoArrowheadTriangle
    
    ' Khối: Prompt D
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 520, 76, 50, 44)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(255, 255, 255)
    shp.Line.ForeColor.RGB = RGB(30, 41, 59)
    shp.TextFrame.TextRange.Text = "Prompt" & vbCrLf & "D"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Font.Size = 8.5
    shp.TextFrame.TextRange.Font.Bold = msoTrue
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    ' Mũi tên từ Prompt D vào Router
    Set shp = sld.Shapes.AddConnector(msoConnectorStraight, 570, 96, 590, 96)
    shp.Line.ForeColor.RGB = RGB(15, 23, 42)
    shp.Line.EndArrowheadStyle = msoArrowheadTriangle
    
    ' Khối: Lightweight Classifier & Dynamic Router
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 590, 46, 130, 114)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(255, 255, 255)
    shp.Line.ForeColor.RGB = RGB(30, 41, 59)
    shp.Line.Weight = 1.2
    shp.TextFrame.TextRange.Text = "Lightweight Classifier" & vbCrLf & _
                                   "& Dynamic Router" & vbCrLf & _
                                   "Input: câu hỏi Q (trong D)" & vbCrLf & _
                                   "A -> Visual / Label reading" & vbCrLf & _
                                   "(hoặc Image-only)" & vbCrLf & _
                                   "B -> Numerical / Arithmetic"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Lines(1).Font.Size = 8
    shp.TextFrame.TextRange.Lines(1).Font.Bold = msoTrue
    shp.TextFrame.TextRange.Lines(2).Font.Size = 8
    shp.TextFrame.TextRange.Lines(2).Font.Bold = msoTrue
    shp.TextFrame.TextRange.Lines(3).Font.Size = 6.8
    shp.TextFrame.TextRange.Lines(3).Font.Color.RGB = RGB(100, 116, 139)
    shp.TextFrame.TextRange.Lines(4).Font.Size = 7
    shp.TextFrame.TextRange.Lines(4).Font.Bold = msoTrue
    shp.TextFrame.TextRange.Lines(4).Font.Color.RGB = RGB(13, 148, 136)
    shp.TextFrame.TextRange.Lines(5).Font.Size = 6.5
    shp.TextFrame.TextRange.Lines(5).Font.Italic = msoTrue
    shp.TextFrame.TextRange.Lines(5).Font.Color.RGB = RGB(13, 148, 136)
    shp.TextFrame.TextRange.Lines(6).Font.Size = 7
    shp.TextFrame.TextRange.Lines(6).Font.Bold = msoTrue
    shp.TextFrame.TextRange.Lines(6).Font.Color.RGB = RGB(194, 65, 12)
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    ' Khung nét đứt cam: VLM Reasoning Engine
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 735, 30, 136, 178)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(255, 255, 255)
    shp.Line.ForeColor.RGB = RGB(245, 158, 11)
    shp.Line.DashStyle = msoLineDash
    shp.Line.Weight = 1.2
    shp.TextFrame.TextRange.Text = "VLM Reasoning Engine" & vbCrLf & "(Qwen2.5-VL-7B)"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Lines(1).Font.Size = 8
    shp.TextFrame.TextRange.Lines(1).Font.Bold = msoTrue
    shp.TextFrame.TextRange.Lines(2).Font.Size = 7.2
    shp.TextFrame.TextRange.Lines(2).Font.Color.RGB = RGB(100, 116, 139)
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    shp.TextFrame.VerticalAnchor = msoAnchorTop
    
    ' (a) Direct Zero-Shot
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 745, 62, 116, 36)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(255, 255, 255)
    shp.Line.ForeColor.RGB = RGB(30, 41, 59)
    shp.TextFrame.TextRange.Text = "(a) Direct Zero-Shot" & vbCrLf & "bypass CoT"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Lines(1).Font.Size = 7.8
    shp.TextFrame.TextRange.Lines(1).Font.Bold = msoTrue
    shp.TextFrame.TextRange.Lines(2).Font.Size = 6.8
    shp.TextFrame.TextRange.Lines(2).Font.Color.RGB = RGB(100, 116, 139)
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    ' (b) Standard CoT (nét đứt)
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 745, 106, 116, 36)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(255, 255, 255)
    shp.Line.ForeColor.RGB = RGB(100, 116, 139)
    shp.Line.DashStyle = msoLineDash
    shp.TextFrame.TextRange.Text = "(b) Standard CoT" & vbCrLf & "baseline (không routing)"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Lines(1).Font.Size = 7.5
    shp.TextFrame.TextRange.Lines(1).Font.Bold = msoTrue
    shp.TextFrame.TextRange.Lines(1).Font.Color.RGB = RGB(71, 85, 105)
    shp.TextFrame.TextRange.Lines(2).Font.Size = 6.8
    shp.TextFrame.TextRange.Lines(2).Font.Color.RGB = RGB(100, 116, 139)
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    ' (c) Adaptive LoRA CoT (nền vàng cam)
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 745, 150, 116, 46)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(254, 243, 199)
    shp.Line.ForeColor.RGB = RGB(217, 119, 6)
    shp.Line.Weight = 1.2
    shp.TextFrame.TextRange.Text = "(c) Adaptive LoRA CoT" & vbCrLf & "3-stage alignment"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Lines(1).Font.Size = 7.8
    shp.TextFrame.TextRange.Lines(1).Font.Bold = msoTrue
    shp.TextFrame.TextRange.Lines(1).Font.Color.RGB = RGB(154, 52, 18)
    shp.TextFrame.TextRange.Lines(2).Font.Size = 6.8
    shp.TextFrame.TextRange.Lines(2).Font.Color.RGB = RGB(154, 52, 18)
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    ' Mũi tên từ Router rẽ vào VLM
    Set shp = sld.Shapes.AddConnector(msoConnectorStraight, 720, 80, 745, 80)
    shp.Line.ForeColor.RGB = RGB(13, 148, 136)
    shp.Line.EndArrowheadStyle = msoArrowheadTriangle
    Set shp = sld.Shapes.AddConnector(msoConnectorStraight, 720, 130, 745, 170)
    shp.Line.ForeColor.RGB = RGB(194, 65, 12)
    shp.Line.EndArrowheadStyle = msoArrowheadTriangle
    
    ' Khối: Unified Output
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 885, 68, 52, 90)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(255, 255, 255)
    shp.Line.ForeColor.RGB = RGB(30, 41, 59)
    shp.TextFrame.TextRange.Text = "Unified Output" & vbCrLf & vbCrLf & "Answer A" & vbCrLf & "(* Rationale R" & vbCrLf & "nếu có)"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Lines(1).Font.Size = 7.5
    shp.TextFrame.TextRange.Lines(1).Font.Bold = msoTrue
    shp.TextFrame.TextRange.Lines(3).Font.Size = 7.5
    shp.TextFrame.TextRange.Lines(3).Font.Bold = msoTrue
    shp.TextFrame.TextRange.Lines(4).Font.Size = 6.5
    shp.TextFrame.TextRange.Lines(4).Font.Color.RGB = RGB(100, 116, 139)
    shp.TextFrame.TextRange.Lines(5).Font.Size = 6.5
    shp.TextFrame.TextRange.Lines(5).Font.Color.RGB = RGB(100, 116, 139)
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    ' Mũi tên hội tụ về Unified Output
    Set shp = sld.Shapes.AddConnector(msoConnectorStraight, 861, 80, 885, 100)
    shp.Line.ForeColor.RGB = RGB(15, 23, 42)
    shp.Line.EndArrowheadStyle = msoArrowheadTriangle
    Set shp = sld.Shapes.AddConnector(msoConnectorStraight, 861, 124, 885, 115)
    shp.Line.ForeColor.RGB = RGB(100, 116, 139)
    shp.Line.DashStyle = msoLineDash
    shp.Line.EndArrowheadStyle = msoArrowheadTriangle
    Set shp = sld.Shapes.AddConnector(msoConnectorStraight, 861, 170, 885, 130)
    shp.Line.ForeColor.RGB = RGB(15, 23, 42)
    shp.Line.EndArrowheadStyle = msoArrowheadTriangle
    
    ' ==========================================================================
    ' 3. PANEL 3 (DƯỚI PHẢI): EVALUATION PIPELINE
    ' ==========================================================================
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 290, 260, 655, 232)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(255, 255, 255)
    shp.Line.ForeColor.RGB = RGB(15, 23, 42)
    shp.Line.Weight = 1.3
    
    ' Header Panel 3
    Set shp = sld.Shapes.AddShape(msoShapeRectangle, 310, 264, 615, 22)
    shp.Fill.Transparency = 1
    shp.Line.Visible = msoFalse
    shp.TextFrame.TextRange.Text = "Adaptive-VLM-Charts: evaluation pipeline"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Font.Size = 10
    shp.TextFrame.TextRange.Font.Bold = msoTrue
    shp.TextFrame.TextRange.Font.Color.RGB = RGB(15, 23, 42)
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    ' Mũi tên từ Unified Output chạy xuống Panel 3
    Set shp = sld.Shapes.AddConnector(msoConnectorStraight, 911, 158, 911, 235)
    shp.Line.ForeColor.RGB = RGB(15, 23, 42)
    Set shp = sld.Shapes.AddConnector(msoConnectorStraight, 911, 235, 340, 235)
    shp.Line.ForeColor.RGB = RGB(15, 23, 42)
    Set shp = sld.Shapes.AddConnector(msoConnectorStraight, 340, 235, 340, 315)
    shp.Line.ForeColor.RGB = RGB(15, 23, 42)
    shp.Line.EndArrowheadStyle = msoArrowheadTriangle
    
    ' Khối: Answer A (+ Rationale R)
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 305, 315, 70, 55)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(255, 255, 255)
    shp.Line.ForeColor.RGB = RGB(30, 41, 59)
    shp.TextFrame.TextRange.Text = "Answer A" & vbCrLf & "(+ Rationale R)"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Lines(1).Font.Size = 8.5
    shp.TextFrame.TextRange.Lines(1).Font.Bold = msoTrue
    shp.TextFrame.TextRange.Lines(2).Font.Size = 7.2
    shp.TextFrame.TextRange.Lines(2).Font.Color.RGB = RGB(100, 116, 139)
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    ' Mũi tên sang Chuẩn hóa
    Set shp = sld.Shapes.AddConnector(msoConnectorStraight, 375, 342, 395, 342)
    shp.Line.ForeColor.RGB = RGB(15, 23, 42)
    shp.Line.EndArrowheadStyle = msoArrowheadTriangle
    
    ' Khối: Chuẩn hóa
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 395, 300, 105, 85)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(255, 255, 255)
    shp.Line.ForeColor.RGB = RGB(30, 41, 59)
    shp.TextFrame.TextRange.Text = "Chuẩn hóa" & vbCrLf & _
                                   "Word-to-Number" & vbCrLf & _
                                   "Bỏ LaTeX" & vbCrLf & _
                                   "Medical Synonyms"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Lines(1).Font.Size = 8.5
    shp.TextFrame.TextRange.Lines(1).Font.Bold = msoTrue
    shp.TextFrame.TextRange.Lines(2).Font.Size = 7.5
    shp.TextFrame.TextRange.Lines(3).Font.Size = 7.5
    shp.TextFrame.TextRange.Lines(4).Font.Size = 7.2
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    ' Mũi tên sang CharXiv & SLAKE
    Set shp = sld.Shapes.AddConnector(msoConnectorStraight, 500, 342, 515, 342)
    shp.Line.ForeColor.RGB = RGB(15, 23, 42)
    Set shp = sld.Shapes.AddConnector(msoConnectorStraight, 515, 342, 515, 315)
    shp.Line.ForeColor.RGB = RGB(15, 23, 42)
    Set shp = sld.Shapes.AddConnector(msoConnectorStraight, 515, 315, 530, 315)
    shp.Line.ForeColor.RGB = RGB(15, 23, 42)
    shp.Line.EndArrowheadStyle = msoArrowheadTriangle
    
    Set shp = sld.Shapes.AddConnector(msoConnectorStraight, 515, 342, 515, 435)
    shp.Line.ForeColor.RGB = RGB(15, 23, 42)
    Set shp = sld.Shapes.AddConnector(msoConnectorStraight, 515, 435, 530, 435)
    shp.Line.ForeColor.RGB = RGB(15, 23, 42)
    shp.Line.EndArrowheadStyle = msoArrowheadTriangle
    
    ' -------------------------------------------------------------
    ' Khung CharXiv: Chấm điểm theo 4 nhóm câu hỏi
    ' -------------------------------------------------------------
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 530, 290, 400, 105)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(255, 255, 255)
    shp.Line.ForeColor.RGB = RGB(30, 41, 59)
    shp.TextFrame.TextRange.Text = "CharXiv (biểu đồ khoa học): chấm điểm theo 4 nhóm câu hỏi"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Font.Size = 8
    shp.TextFrame.TextRange.Font.Bold = msoTrue
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    shp.TextFrame.VerticalAnchor = msoAnchorTop
    
    ' match_fixed / Bootstrap 95% CI
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 540, 316, 110, 68)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(255, 255, 255)
    shp.Line.ForeColor.RGB = RGB(30, 41, 59)
    shp.TextFrame.TextRange.Text = "match_fixed" & vbCrLf & "Bootstrap 95% CI"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Lines(1).Font.Size = 8
    shp.TextFrame.TextRange.Lines(1).Font.Bold = msoTrue
    shp.TextFrame.TextRange.Lines(2).Font.Size = 7.2
    shp.TextFrame.TextRange.Lines(2).Font.Color.RGB = RGB(100, 116, 139)
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    ' Mũi tên từ match_fixed sang 4 nhóm
    Set shp = sld.Shapes.AddConnector(msoConnectorStraight, 650, 350, 670, 350)
    shp.Line.ForeColor.RGB = RGB(15, 23, 42)
    shp.Line.EndArrowheadStyle = msoArrowheadTriangle
    
    ' 4 ô con nhóm câu hỏi: Text-in-Chart, Text-in-General, Number-in-Chart, Number-in-General
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 675, 318, 115, 30)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(255, 255, 255)
    shp.Line.ForeColor.RGB = RGB(30, 41, 59)
    shp.TextFrame.TextRange.Text = "Text-in-Chart"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Font.Size = 7.5
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 800, 318, 115, 30)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(255, 255, 255)
    shp.Line.ForeColor.RGB = RGB(30, 41, 59)
    shp.TextFrame.TextRange.Text = "Text-in-General"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Font.Size = 7.5
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 675, 354, 115, 30)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(255, 255, 255)
    shp.Line.ForeColor.RGB = RGB(30, 41, 59)
    shp.TextFrame.TextRange.Text = "Number-in-Chart"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Font.Size = 7.5
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 800, 354, 115, 30)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(255, 255, 255)
    shp.Line.ForeColor.RGB = RGB(30, 41, 59)
    shp.TextFrame.TextRange.Text = "Number-in-General"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Font.Size = 7.5
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    ' -------------------------------------------------------------
    ' Khung SLAKE: Accuracy & NLG
    ' -------------------------------------------------------------
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 530, 404, 400, 78)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(255, 255, 255)
    shp.Line.ForeColor.RGB = RGB(30, 41, 59)
    shp.TextFrame.TextRange.Text = "SLAKE (Y khoa): Accuracy & NLG"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Font.Size = 8
    shp.TextFrame.TextRange.Font.Bold = msoTrue
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    shp.TextFrame.VerticalAnchor = msoAnchorTop
    
    ' 3 ô con SLAKE: Closed, Open NLG, Open
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 540, 426, 120, 46)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(255, 255, 255)
    shp.Line.ForeColor.RGB = RGB(30, 41, 59)
    shp.TextFrame.TextRange.Text = "Closed:" & vbCrLf & "Strict Acc, Medical Acc"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Lines(1).Font.Size = 7.5
    shp.TextFrame.TextRange.Lines(1).Font.Bold = msoTrue
    shp.TextFrame.TextRange.Lines(2).Font.Size = 6.8
    shp.TextFrame.TextRange.Lines(2).Font.Color.RGB = RGB(51, 65, 85)
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 670, 426, 130, 46)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(255, 255, 255)
    shp.Line.ForeColor.RGB = RGB(30, 41, 59)
    shp.TextFrame.TextRange.Text = "Open NLG:" & vbCrLf & "BLEU-1, 2 | ROUGE-1, L"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Lines(1).Font.Size = 7.5
    shp.TextFrame.TextRange.Lines(1).Font.Bold = msoTrue
    shp.TextFrame.TextRange.Lines(2).Font.Size = 6.8
    shp.TextFrame.TextRange.Lines(2).Font.Color.RGB = RGB(51, 65, 85)
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    Set shp = sld.Shapes.AddShape(msoShapeRoundedRectangle, 810, 426, 110, 46)
    shp.Fill.Solid
    shp.Fill.ForeColor.RGB = RGB(255, 255, 255)
    shp.Line.ForeColor.RGB = RGB(30, 41, 59)
    shp.TextFrame.TextRange.Text = "Open:" & vbCrLf & "Strict EM, Medical Sem. Acc"
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Lines(1).Font.Size = 7.5
    shp.TextFrame.TextRange.Lines(1).Font.Bold = msoTrue
    shp.TextFrame.TextRange.Lines(2).Font.Size = 6.4
    shp.TextFrame.TextRange.Lines(2).Font.Color.RGB = RGB(51, 65, 85)
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    ' ==========================================================================
    ' 4. CAPTION BÀI BÁO TẠP CHÍ Ở DƯỚI CÙNG
    ' ==========================================================================
    Set shp = sld.Shapes.AddShape(msoShapeRectangle, 15, 502, 930, 32)
    shp.Fill.Transparency = 1
    shp.Line.Visible = msoFalse
    shp.TextFrame.TextRange.Text = "Hình 1. Tổng quan Adaptive-VLM-Charts. Đầu vào gồm ảnh I và/hoặc câu hỏi Q (nét đứt: tùy chọn); đầu vào chỉ có Q được truy hồi ảnh qua Hybrid RAG. " & _
                                   "Router nhận văn bản câu hỏi Q trong prompt D: câu hỏi số học -> Adaptive LoRA CoT; câu hỏi đọc nhãn hoặc đầu vào chỉ có ảnh -> Direct Zero-Shot " & _
                                   "(Standard CoT là baseline; đường Oracle / Direct Bypass là thiết lập cận trên). Đầu ra thống nhất được chuẩn hóa rồi chấm điểm match_fixed (CharXiv) " & _
                                   "và BLEU, ROUGE (SLAKE); Adaptive Dynamic Routing đạt 39.60% (CharXiv) và 52.59% (SLAKE)."
    shp.TextFrame.TextRange.Font.Name = "Arial"
    shp.TextFrame.TextRange.Font.Size = 6.8
    shp.TextFrame.TextRange.Font.Color.RGB = RGB(30, 41, 59)
    shp.TextFrame.TextRange.ParagraphFormat.Alignment = ppAlignCenter
    
    MsgBox "Đã vẽ hoàn tất sơ đồ kiến trúc Adaptive-VLM-Charts chuẩn 1:1 theo ảnh mẫu vào Slide PowerPoint!", vbInformation, "Thành Công"
End Sub
