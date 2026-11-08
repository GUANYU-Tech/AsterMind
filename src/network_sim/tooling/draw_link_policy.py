"""Render a local architecture diagram without executing browser content."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def main():
    image = Image.new("RGB", (1600, 1000), "white")
    draw = ImageDraw.Draw(image)
    normal = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 27)
    small = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 23)
    title = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", 35)
    draw.text((70, 35), "动态建链：候选连接 → 终端预留 → 可用通路 → 传输反馈", font=title, fill="#23304a")

    def box(x, y, w, h, lines, color="#eaf0ff"):
        draw.rounded_rectangle((x,y,x+w,y+h), radius=18, fill=color, outline="#889fbd", width=2)
        for index, text in enumerate(lines):
            draw.text((x+w/2, y+28+index*38), text, font=normal if index == 0 else small,
                      fill="#23304a", anchor="mt")

    def arrow(points):
        draw.line(points, fill="#768ba8", width=4)
        x,y=points[-1]; ax,ay=points[-2]
        if x > ax:
            tri=[(x,y),(x-14,y-9),(x-14,y+9)]
        elif x < ax:
            tri=[(x,y),(x+14,y-9),(x+14,y+9)]
        else:
            tri=[(x,y),(x-9,y-14),(x+9,y-14)]
        draw.polygon(tri,fill="#768ba8")

    box(60,135,440,145,["物理配置 · topology.py", "Cell / 卫星 / GPU / Switch / 终端", "候选边、容量、延迟"])
    box(1100,135,440,145,["外部可用窗口", "当前用合成窗口验证", "未来由 Orbit 输出可见性"])
    box(560,335,480,175,["建链控制 · link_policy.py", "保持旧链，空闲终端选伙伴", "捕获占名额；完成后启用链路", "O-S 选择规则作另一参考版本"], "#fff4e6")
    arrow([(280,280),(280,370),(560,370)])
    arrow([(1320,280),(1320,370),(1040,370)])
    box(60,580,440,140,["上层点对点任务", "起终点 / 字节数 / 就绪时间", "多 GPU 操作层、O-S、M-L"])
    box(560,580,480,140,["传输执行 · runtime.py", "FlowSim 路径与共享带宽", "无可用路径的未开始任务等待"], "#eaf6f1")
    box(1100,580,440,140,["全局事件时钟", "next_event_time_ns", "advance_to(time_ns)"], "#fff4e6")
    arrow([(800,510),(800,580)])
    arrow([(500,650),(560,650)])
    arrow([(1100,650),(1040,650)])
    box(560,800,480,140,["成功 / 失败反馈", "数据到达 → 推进依赖计算", "路径断链 → 上层决定如何重试"], "#eaf6f1")
    arrow([(800,720),(800,800)])
    draw.text((60,810), "三项时间可配：", font=normal, fill="#23304a")
    draw.text((60,860), "检查间隔、初次、重新捕获", font=small, fill="#65738a")
    draw.text((1100,810), "当前边界：", font=normal, fill="#23304a")
    draw.text((1100,860), "已知断链会终止在途任务", font=small, fill="#65738a")
    draw.text((1100,900), "自动换路、数据续传待实现", font=small, fill="#65738a")
    image.save(ROOT / "docs/link_policy.png")


if __name__ == "__main__":
    main()
