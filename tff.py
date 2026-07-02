from matplotlib import font_manager

# 获取所有可用字体名称
font_list = [f.name for f in font_manager.fontManager.ttflist]
# 检查是否包含Times New Roman
is_times_roman = "Times New Roman" in font_list or "Times_New_Roman" in font_list

if is_times_roman:
    print("✅ 已安装Times New Roman字体")
else:
    print("❌ 未安装Times New Roman字体")