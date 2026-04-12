
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider
import matplotlib
import numpy as np
import os

# 配置 matplotlib 支持中文显示
# 尝试检测可用的中文字体
import matplotlib.font_manager as fm

# 获取系统中所有可用字体
available_fonts = [f.name for f in fm.fontManager.ttflist]

# 按优先级选择中文字体
chinese_fonts = []
font_candidates = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong', 'STHeiti', 'STSong']
for font in font_candidates:
    if font in available_fonts:
        chinese_fonts.append(font)
        break

# 如果没有找到中文字体，使用默认字体（可能会显示警告，但不影响功能）
if not chinese_fonts:
    chinese_fonts = ['DejaVu Sans']
    print("警告：未找到中文字体，中文可能无法正确显示。建议安装SimHei或Microsoft YaHei字体。")

# 设置字体（添加DejaVu Sans作为最后的回退）
font_list = chinese_fonts + ['DejaVu Sans']
plt.rcParams['font.sans-serif'] = font_list
plt.rcParams['axes.unicode_minus'] = False

# 设置matplotlib全局配置，确保保存图片时也能正确显示中文
matplotlib.rcParams['font.sans-serif'] = font_list
matplotlib.rcParams['axes.unicode_minus'] = False


def create_rsrp_plot(plot_data, distance_col, rsrp_col, 
                     min_dist, max_dist, title, 
                     label_text=None, color='tab:blue',
                     figsize_width=None, show_slider=True):
    """
    创建RSRP图表（重写版本）
    
    Args:
        plot_data: 要绘制的数据DataFrame
        distance_col: 距离列名
        rsrp_col: RSRP列名
        min_dist: 数据的最小距离值
        max_dist: 数据的最大距离值
        title: 图表标题
        label_text: 图例标签文本
        color: 线条颜色
        figsize_width: 图表宽度（如果为None则自动计算）
        show_slider: 是否显示缩放滑块
        
    Returns:
        fig, ax: matplotlib图表对象
    """
    # 计算距离范围
    distance_range = max_dist - min_dist
    
    # 根据数据范围动态计算图表宽度
    if figsize_width is None:
        if distance_range > 0:
            # 每1km对应约0.4英寸，最小10英寸，最大14英寸
            figsize_width = max(10, min(14, 8 + distance_range * 0.4))
        else:
            figsize_width = 10
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(figsize_width, 7))
    
    # 紧凑布局：最小化边距
    if show_slider:
        plt.subplots_adjust(left=0.1, bottom=0.18, right=0.95, top=0.9)
    else:
        plt.subplots_adjust(left=0.1, bottom=0.12, right=0.95, top=0.9)
    
    # 绘制数据曲线
    ax.plot(plot_data[distance_col], 
            plot_data[rsrp_col],
            color=color,
            linewidth=2,
            marker='o',
            markersize=3,
            alpha=0.8,
            label=label_text if label_text else 'RSRP')
    
    # 设置横轴范围 - 严格匹配数据范围，只添加很小的边距
    margin = max(distance_range * 0.01, 0.01)  # 1%边距或最小0.01km
    x_min = min_dist - margin
    x_max = max_dist + margin
    ax.set_xlim(x_min, x_max)
    
    # 设置横轴刻度 - 根据距离范围智能调整
    if distance_range > 50:
        step = 10
    elif distance_range > 20:
        step = 5
    elif distance_range > 10:
        step = 2
    elif distance_range > 5:
        step = 1
    elif distance_range > 2:
        step = 0.5
    else:
        step = 0.2
    
    # 生成刻度位置 - 从实际数据范围开始
    tick_start = np.floor(min_dist / step) * step
    tick_end = np.ceil(max_dist / step) * step
    x_ticks = np.arange(tick_start, tick_end + step, step)
    # 过滤掉超出范围的刻度
    x_ticks = x_ticks[(x_ticks >= x_min) & (x_ticks <= x_max)]
    ax.set_xticks(x_ticks)
    ax.tick_params(axis='x', rotation=45)
    
    # 设置图表属性
    ax.set_xlabel('公里标 (km)', fontsize=12)
    ax.set_ylabel('RSRP (dBm)', fontsize=12)
    ax.set_title(title, fontsize=14)
    if label_text:
        ax.legend(loc='best', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 添加RSRP信号强度参考线
    ax.axhline(y=-80, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=-90, color='orange', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=-100, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    # 添加参考线标签
    ax.text(x_max * 0.995, -80, '良好 (-80 dBm)', 
            verticalalignment='bottom', horizontalalignment='right',
            color='green', fontsize=9, alpha=0.7)
    ax.text(x_max * 0.995, -90, '中等 (-90 dBm)', 
            verticalalignment='bottom', horizontalalignment='right',
            color='orange', fontsize=9, alpha=0.7)
    ax.text(x_max * 0.995, -100, '较差 (-100 dBm)', 
            verticalalignment='bottom', horizontalalignment='right',
            color='red', fontsize=9, alpha=0.7)
    
    # 创建横轴缩放滑块（如果启用）
    if show_slider:
        slider_ax = fig.add_axes([0.1, 0.02, 0.85, 0.03])
        range_slider = RangeSlider(
            slider_ax,
            '距离范围 (km)',
            min_dist,  # slider的最小值就是数据的最小值
            max_dist,  # slider的最大值就是数据的最大值
            valinit=(x_min, x_max),
            valfmt='%.2f'
        )
        
        def update_xlim(val):
            """当slider值改变时更新横轴范围"""
            min_val, max_val = val
            ax.set_xlim(min_val, max_val)
            
            # 更新横轴刻度
            current_range = max_val - min_val
            if current_range > 50:
                step = 10
            elif current_range > 20:
                step = 5
            elif current_range > 10:
                step = 2
            elif current_range > 5:
                step = 1
            elif current_range > 2:
                step = 0.5
            else:
                step = 0.2
            
            tick_start = np.floor(min_val / step) * step
            tick_end = np.ceil(max_val / step) * step
            x_ticks = np.arange(tick_start, tick_end + step, step)
            x_ticks = x_ticks[(x_ticks >= min_val) & (x_ticks <= max_val)]
            ax.set_xticks(x_ticks)
            fig.canvas.draw_idle()
        
        range_slider.on_changed(update_xlim)
        fig.text(0.5, 0.1, '拖动滑块调整横轴显示范围', 
                 ha='center', fontsize=10, style='italic', alpha=0.7)
    
    return fig, ax


# 读取Excel文件
def read_and_plot_rsrp(file_path=None, max_points_per_station=5000, aggregate_bin_size=0.01):
    """
    读取并绘制RSRP数据
    
    Args:
        file_path: Excel文件路径（如果为None，则自动在脚本目录下查找nr_info.xlsx）
        max_points_per_station: 每个基站最大显示点数（如果超过则采样）
        aggregate_bin_size: 数据聚合的区间大小（km），如果为None则不聚合
    """
    # 如果未提供文件路径，自动查找
    if file_path is None:
        # 获取脚本所在目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # 在脚本目录下查找nr_info.xlsx
        file_path = os.path.join(script_dir, 'nr_info.xlsx')
        if not os.path.exists(file_path):
            # 如果脚本目录下没有，尝试在父目录查找
            parent_dir = os.path.dirname(script_dir)
            file_path = os.path.join(parent_dir, 'nr_info.xlsx')
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"未找到nr_info.xlsx文件！\n"
                f"已尝试查找位置：\n"
                f"  1. {os.path.join(script_dir, 'nr_info.xlsx')}\n"
                f"  2. {os.path.join(parent_dir, 'nr_info.xlsx')}\n"
                f"请确保文件存在或提供正确的文件路径。"
            )
    
    # 如果提供的是相对路径，转换为绝对路径
    if not os.path.isabs(file_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # 先尝试在脚本目录下查找
        abs_path = os.path.join(script_dir, file_path)
        if not os.path.exists(abs_path):
            # 如果脚本目录下没有，尝试在当前工作目录查找
            abs_path = os.path.abspath(file_path)
        file_path = abs_path
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    print(f"读取文件: {file_path}")
    
    # 读取Excel文件
    df = pd.read_excel(file_path, sheet_name='Sheet1')
    
    # 清理列名（去除可能的空格）
    df.columns = df.columns.str.strip()
    
    # 提取需要的列：公里标（距离）、SSS RSRP(dBm)（信号强度）、Cell ID（小区ID，用于区分基站）
    # 根据数据，使用 Cell ID 来区分不同基站
    distance_col = '公里标(km)'
    rsrp_col = 'SSS RSRP(dBm)'
    cell_id_col = 'Cell ID'
    gnodeb_col = 'gNodeB ID'
    
    # 创建一个组合标识来区分不同的基站/小区
    df['基站标识'] = df[gnodeb_col].astype(str) + '_Cell' + df[cell_id_col].astype(str)
    
    # 获取所有唯一的基站标识
    unique_stations = df['基站标识'].unique()
    
    # 设置颜色映射
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_stations)))
    color_map = dict(zip(unique_stations, colors))
    
    # 计算全局距离范围（用于设置横轴范围）
    global_min_dist = df[distance_col].min()
    global_max_dist = df[distance_col].max()
    distance_range = global_max_dist - global_min_dist
    
    print(f"数据距离范围: {distance_range:.2f} km (从 {global_min_dist:.2f} 到 {global_max_dist:.2f})")
    
    # 准备所有基站的数据
    all_plot_data = []
    for station in unique_stations:
        station_data = df[df['基站标识'] == station].copy()
        station_data = station_data.sort_values(by=distance_col)
        
        # 如果数据点太多，进行采样或聚合
        if len(station_data) > max_points_per_station:
            if aggregate_bin_size is not None:
                station_data['距离区间'] = (station_data[distance_col] / aggregate_bin_size).round() * aggregate_bin_size
                aggregated = station_data.groupby('距离区间').agg({
                    distance_col: 'mean',
                    rsrp_col: 'mean'
                }).reset_index()
                plot_data = aggregated.sort_values(by=distance_col)
                print(f"  基站 {station}: {len(station_data)} 个数据点 -> 聚合为 {len(plot_data)} 个点")
            else:
                indices = np.linspace(0, len(station_data) - 1, max_points_per_station, dtype=int)
                plot_data = station_data.iloc[indices].copy()
                print(f"  基站 {station}: {len(station_data)} 个数据点 -> 采样为 {len(plot_data)} 个点")
        else:
            plot_data = station_data
        
        all_plot_data.append((station, plot_data))
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 7))
    plt.subplots_adjust(left=0.1, bottom=0.18, right=0.95, top=0.9)
    
    # 绘制所有基站的曲线
    for station, plot_data in all_plot_data:
        ax.plot(plot_data[distance_col], 
                plot_data[rsrp_col], 
                color=color_map[station],
                label=f'基站 {station}',
                linewidth=1.5,
                marker='o',
                markersize=2,
                alpha=0.8)
    
    # 设置横轴范围 - 严格匹配数据范围
    margin = max(distance_range * 0.01, 0.01)
    x_min = global_min_dist - margin
    x_max = global_max_dist + margin
    ax.set_xlim(x_min, x_max)
    print(f"横轴范围设置为: {x_min:.3f} - {x_max:.3f} km")
    
    # 设置横轴刻度
    if distance_range > 50:
        step = 10
    elif distance_range > 20:
        step = 5
    elif distance_range > 10:
        step = 2
    elif distance_range > 5:
        step = 1
    elif distance_range > 2:
        step = 0.5
    else:
        step = 0.2
    
    tick_start = np.floor(global_min_dist / step) * step
    tick_end = np.ceil(global_max_dist / step) * step
    x_ticks = np.arange(tick_start, tick_end + step, step)
    x_ticks = x_ticks[(x_ticks >= x_min) & (x_ticks <= x_max)]
    ax.set_xticks(x_ticks)
    ax.tick_params(axis='x', rotation=45)
    
    # 设置图表属性
    ax.set_xlabel('公里标 (km)', fontsize=12)
    ax.set_ylabel('RSRP (dBm)', fontsize=12)
    ax.set_title('铁路沿线5G信号RSRP强度随距离变化图', fontsize=14)
    ax.legend(loc='best', fontsize=9, ncol=2 if len(unique_stations) > 5 else 1)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 添加RSRP信号强度参考线
    ax.axhline(y=-80, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=-90, color='orange', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=-100, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    ax.text(x_max * 0.995, -80, '良好 (-80 dBm)', 
            verticalalignment='bottom', horizontalalignment='right',
            color='green', fontsize=9, alpha=0.7)
    ax.text(x_max * 0.995, -90, '中等 (-90 dBm)', 
            verticalalignment='bottom', horizontalalignment='right',
            color='orange', fontsize=9, alpha=0.7)
    ax.text(x_max * 0.995, -100, '较差 (-100 dBm)', 
            verticalalignment='bottom', horizontalalignment='right',
            color='red', fontsize=9, alpha=0.7)
    
    # 创建横轴缩放滑块
    slider_ax = fig.add_axes([0.1, 0.02, 0.85, 0.03])
    range_slider = RangeSlider(
        slider_ax,
        '距离范围 (km)',
        global_min_dist,  # slider范围从实际数据的最小值开始
        global_max_dist,  # slider范围到实际数据的最大值结束
        valinit=(x_min, x_max),
        valfmt='%.2f'
    )
    
    def update_xlim(val):
        min_val, max_val = val
        ax.set_xlim(min_val, max_val)
        current_range = max_val - min_val
        if current_range > 50:
            step = 10
        elif current_range > 20:
            step = 5
        elif current_range > 10:
            step = 2
        elif current_range > 5:
            step = 1
        elif current_range > 2:
            step = 0.5
        else:
            step = 0.2
        tick_start = np.floor(min_val / step) * step
        tick_end = np.ceil(max_val / step) * step
        x_ticks = np.arange(tick_start, tick_end + step, step)
        x_ticks = x_ticks[(x_ticks >= min_val) & (x_ticks <= max_val)]
        ax.set_xticks(x_ticks)
        fig.canvas.draw_idle()
    
    range_slider.on_changed(update_xlim)
    fig.text(0.5, 0.1, '拖动滑块调整横轴显示范围', 
             ha='center', fontsize=10, style='italic', alpha=0.7)
    
    # 保存图片到脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'rsrp_distance_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存至: {output_path}")
    print("提示：在显示的图表窗口中，可以使用底部的滑块调整横轴显示范围")
    plt.show()
    
    # 打印统计信息
    print("\n=== 数据统计信息 ===")
    print(f"数据点总数: {len(df)}")
    print(f"距离范围: {global_min_dist:.2f} km - {global_max_dist:.2f} km (总长度: {distance_range:.2f} km)")
    print(f"RSRP范围: {df[rsrp_col].min():.2f} dBm - {df[rsrp_col].max():.2f} dBm")
    print(f"\n各基站数据点数:")
    for station in unique_stations:
        count = len(df[df['基站标识'] == station])
        print(f"  {station}: {count} 个数据点")


def plot_by_pci(file_path=None, max_points_per_station=5000, aggregate_bin_size=0.01):
    """
    根据PCI字段分离数据，为每个基站生成单独的图表
    
    Args:
        file_path: Excel文件路径（如果为None，则自动在脚本目录下查找nr_info.xlsx）
        max_points_per_station: 每个基站最大显示点数（如果超过则采样）
        aggregate_bin_size: 数据聚合的区间大小（km），如果为None则不聚合
    """
    # 如果未提供文件路径，自动查找
    if file_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, 'nr_info.xlsx')
        if not os.path.exists(file_path):
            parent_dir = os.path.dirname(script_dir)
            file_path = os.path.join(parent_dir, 'nr_info.xlsx')
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"未找到nr_info.xlsx文件！\n"
                f"已尝试查找位置：\n"
                f"  1. {os.path.join(script_dir, 'nr_info.xlsx')}\n"
                f"  2. {os.path.join(parent_dir, 'nr_info.xlsx')}\n"
                f"请确保文件存在或提供正确的文件路径。"
            )
    
    if not os.path.isabs(file_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        abs_path = os.path.join(script_dir, file_path)
        if not os.path.exists(abs_path):
            abs_path = os.path.abspath(file_path)
        file_path = abs_path
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    print(f"读取文件: {file_path}")
    
    # 读取Excel文件
    df = pd.read_excel(file_path, sheet_name='Sheet1')
    df.columns = df.columns.str.strip()
    
    # 检查PCI字段是否存在
    pci_col = None
    for col in df.columns:
        if 'PCI' in col.upper():
            pci_col = col
            break
    
    if pci_col is None:
        raise ValueError("未找到PCI字段！请检查Excel文件中是否包含PCI列。")
    
    print(f"使用字段 '{pci_col}' 作为PCI标识")
    
    # 提取需要的列
    distance_col = '公里标(km)'
    rsrp_col = 'SSS RSRP(dBm)'
    
    # 检查必要的列是否存在
    required_cols = [distance_col, rsrp_col, pci_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要的列: {missing_cols}")
    
    # 获取所有唯一的PCI值
    unique_pcis = df[pci_col].dropna().unique()
    unique_pcis = sorted([pci for pci in unique_pcis if pd.notna(pci)])
    
    print(f"\n找到 {len(unique_pcis)} 个不同的PCI值: {unique_pcis}")
    
    # 获取脚本所在目录（保存路径）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 为每个PCI生成单独的图表
    for pci in unique_pcis:
        print(f"\n处理 PCI={pci}...")
        
        # 筛选该PCI的数据
        pci_data = df[df[pci_col] == pci].copy()
        pci_data = pci_data.sort_values(by=distance_col)
        
        if len(pci_data) == 0:
            print(f"  PCI={pci} 没有数据，跳过")
            continue
        
        # 计算距离范围 - 使用实际数据的最小值和最大值
        global_min_dist = pci_data[distance_col].min()
        global_max_dist = pci_data[distance_col].max()
        distance_range = global_max_dist - global_min_dist
        
        print(f"  PCI {pci} 数据距离范围: {distance_range:.2f} km (从 {global_min_dist:.2f} 到 {global_max_dist:.2f})")
        
        # 数据采样或聚合
        if len(pci_data) > max_points_per_station:
            if aggregate_bin_size is not None:
                pci_data['距离区间'] = (pci_data[distance_col] / aggregate_bin_size).round() * aggregate_bin_size
                aggregated = pci_data.groupby('距离区间').agg({
                    distance_col: 'mean',
                    rsrp_col: 'mean'
                }).reset_index()
                plot_data = aggregated.sort_values(by=distance_col)
                print(f"    {len(pci_data)} 个数据点 -> 聚合为 {len(plot_data)} 个点")
            else:
                indices = np.linspace(0, len(pci_data) - 1, max_points_per_station, dtype=int)
                plot_data = pci_data.iloc[indices].copy()
                print(f"    {len(pci_data)} 个数据点 -> 采样为 {len(plot_data)} 个点")
        else:
            plot_data = pci_data
        
        # 根据数据范围动态计算图表宽度
        if distance_range > 0:
            fig_width = max(10, min(14, 8 + distance_range * 0.4))
        else:
            fig_width = 10
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(fig_width, 7))
        plt.subplots_adjust(left=0.1, bottom=0.18, right=0.95, top=0.9)
        
        # 绘制RSRP曲线
        ax.plot(plot_data[distance_col], 
                plot_data[rsrp_col], 
                color='tab:blue',
                linewidth=2,
                marker='o',
                markersize=3,
                alpha=0.8,
                label=f'PCI {pci}')
        
        # 设置横轴范围 - 严格匹配数据范围（从实际数据的最小值开始）
        margin = max(distance_range * 0.01, 0.01)
        x_min = global_min_dist - margin
        x_max = global_max_dist + margin
        ax.set_xlim(x_min, x_max)
        print(f"    横轴范围设置为: {x_min:.3f} - {x_max:.3f} km")
        
        # 设置横轴刻度 - 从实际数据范围开始
        if distance_range > 50:
            step = 10
        elif distance_range > 20:
            step = 5
        elif distance_range > 10:
            step = 2
        elif distance_range > 5:
            step = 1
        elif distance_range > 2:
            step = 0.5
        else:
            step = 0.2
        
        # 刻度从实际数据的最小值开始，而不是从0开始
        tick_start = np.floor(global_min_dist / step) * step
        tick_end = np.ceil(global_max_dist / step) * step
        x_ticks = np.arange(tick_start, tick_end + step, step)
        # 只保留在显示范围内的刻度
        x_ticks = x_ticks[(x_ticks >= x_min) & (x_ticks <= x_max)]
        ax.set_xticks(x_ticks)
        ax.tick_params(axis='x', rotation=45)
        
        # 设置图表属性
        ax.set_xlabel('公里标 (km)', fontsize=12)
        ax.set_ylabel('RSRP (dBm)', fontsize=12)
        ax.set_title(f'PCI {pci} - 铁路沿线5G信号RSRP强度随距离变化图', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 添加RSRP信号强度参考线
        ax.axhline(y=-80, color='green', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=-90, color='orange', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=-100, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        ax.text(x_max * 0.995, -80, '良好 (-80 dBm)', 
                verticalalignment='bottom', horizontalalignment='right',
                color='green', fontsize=9, alpha=0.7)
        ax.text(x_max * 0.995, -90, '中等 (-90 dBm)', 
                verticalalignment='bottom', horizontalalignment='right',
                color='orange', fontsize=9, alpha=0.7)
        ax.text(x_max * 0.995, -100, '较差 (-100 dBm)', 
                verticalalignment='bottom', horizontalalignment='right',
                color='red', fontsize=9, alpha=0.7)
        
        # 创建横轴缩放滑块 - slider范围从实际数据的最小值开始
        slider_ax = fig.add_axes([0.1, 0.02, 0.85, 0.03])
        range_slider = RangeSlider(
            slider_ax,
            '距离范围 (km)',
            global_min_dist,  # slider最小值 = 数据最小值（不是0）
            global_max_dist,  # slider最大值 = 数据最大值
            valinit=(x_min, x_max),
            valfmt='%.2f'
        )
        
        def update_xlim(val):
            min_val, max_val = val
            ax.set_xlim(min_val, max_val)
            current_range = max_val - min_val
            if current_range > 50:
                step = 10
            elif current_range > 20:
                step = 5
            elif current_range > 10:
                step = 2
            elif current_range > 5:
                step = 1
            elif current_range > 2:
                step = 0.5
            else:
                step = 0.2
            tick_start = np.floor(min_val / step) * step
            tick_end = np.ceil(max_val / step) * step
            x_ticks = np.arange(tick_start, tick_end + step, step)
            x_ticks = x_ticks[(x_ticks >= min_val) & (x_ticks <= max_val)]
            ax.set_xticks(x_ticks)
            fig.canvas.draw_idle()
        
        range_slider.on_changed(update_xlim)
        fig.text(0.5, 0.1, '拖动滑块调整横轴显示范围', 
                 ha='center', fontsize=10, style='italic', alpha=0.7)
        
        # 保存图表
        output_filename = f'rsrp_pci_{pci}_plot.png'
        output_path = os.path.join(script_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  图表已保存至: {output_path}")
        plt.close()  # 关闭图表以释放内存
    
    print(f"\n=== 处理完成 ===")
    print(f"共生成 {len(unique_pcis)} 个图表")
    print(f"保存路径: {script_dir}")


# 如果直接运行此脚本
if __name__ == "__main__":
    # 参数说明：
    # file_path: Excel文件路径（如果为None，会自动在脚本目录下查找nr_info.xlsx）
    # max_points_per_station: 每个基站最大显示点数，超过则进行采样或聚合（默认5000）
    # aggregate_bin_size: 数据聚合区间大小（km），如果为None则使用采样，否则使用聚合（默认0.01km=10m）
    #   设置为None时，如果数据点超过max_points_per_station，会均匀采样
    #   设置为数值时，会按距离区间聚合（取平均值），可以更好地保留趋势
    
    # ===== 选择运行模式 =====
    
    # 模式1：按PCI分离，为每个基站生成单独图表（推荐）
    plot_by_pci(file_path=None, max_points_per_station=5000, aggregate_bin_size=0.01)
    
    # 模式2：所有基站在一个图表中显示
    # read_and_plot_rsrp(file_path=None, max_points_per_station=5000, aggregate_bin_size=0.01)
    
    # 方式3：手动指定文件路径（如果需要）
    # plot_by_pci(file_path="nr_info.xlsx", max_points_per_station=5000, aggregate_bin_size=0.01)
    # read_and_plot_rsrp(file_path="nr_info.xlsx", max_points_per_station=5000, aggregate_bin_size=0.01)
