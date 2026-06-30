车牌标注参考（检测与识别分开）
================================
本项目车牌流程: YOLO 框车牌(可训练) + HyperLPR3 读字(预训练，无需训练)

一、检测标注（训练 YOLO）
  工具: LabelImg，格式 YOLO
  目录: reference/images + reference/labels
  类别: plate (class_id=0)
  每行: 0 cx cy w h (归一化 0~1)
  模板: templates/detection_label_example.txt

二、识别标注（OCR 评测/核对，不用于训练）
  目录: reference/recognition/*.json
  与检测框一一对应，填写 plate_text 真值
  模板: templates/recognition_label_example.json

三、新增数据
  images/train + labels/train + recognition/train (可选)
  对照 preview/ 红框与绿字核对

四、评测 OCR 准确率
  python scripts/eval_plate_recognition.py --split val
