# cpu/ — chạy trên máy thường, không cần GPU

Dữ liệu (`results/`, `raw_runs/`) nằm ở thư mục gốc repo, các script tự tìm thấy
(ghi đè bằng `MATHKT_ROOT` nếu để chỗ khác). Chạy tất cả: `bash ../reproduce.sh`.

    python3 01_judge_threeway.py          # đã chạy -> out_01_*.json
    python3 02_calibration_downstream.py  # đã chạy -> out_02_*.json
    python3 03_table1_cis.py              # đã chạy -> out_03_*.csv  (~15s)
    python3 04_ea_confusion_table.py      # đã chạy -> out_04_*.csv/.tex
    python3 05_regen_automatic_checks.py  # đã chạy -> out_05_*.csv
    python3 09_build_atc_graph.py         # đã chạy -> ../out/prereq_graph_mathdial_atc.json (= notebook A)
    python3 08_endtoend_mta.py            # đã chạy -> out_08_*.csv/.json (mặc định: ../out/prereq_graph_mathdial_atc.json, dựng bằng quy tắc)

Cần thêm đầu vào:

    python3 06_score_human_verification.py --merge   # sau khi 2 người gán nhãn 139 lượt
    python3 06_score_human_verification.py           # sau khi thống nhất
    python3 07_score_feedback_baseline.py            # sau Colab B + chấm 60 mục
