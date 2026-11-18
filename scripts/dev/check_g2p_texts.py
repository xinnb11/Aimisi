import sys
from pathlib import Path
sys.path.insert(0, r"D:\语音模型项目\scripts")
import verify_onnx_tts as V

MODEL = Path(r"D:\爱弥斯\_work\publish")
frontend = V.TextFrontend(MODEL)

texts = {
    "平静": "同桌，你好。我是拉贝尔学部的爱弥斯，和你一样。",
    "调侃": "啊！刚才没来得及说。那时候，您勇救莫宁教授的身姿，非常酷哦！",
    "温柔": "在那些听到你叹息的夜晚，我悄悄走到桌子旁时，偶尔会听到你在和他低声通讯。守岸人，我记得你念起的这个名字。每当你和他说话的时候，那些叹息声都会少一些。我想，他应该就是那个一直守望着你的平静与安宁的人吧。",
    "愤怒生气": "知道那些真相的时候，我感受到的第一种情绪，是愤怒。和我一起生活的记忆越是温暖，我就越是憎恨那些让你痛苦的存在。我恨不得用你教我的剑法，把那些事物都斩碎。我只是作为幽灵飘荡了十几年，就已经时常觉得寂寞了。而你……那些流淌过你的漫长时间……一个人从孤独中来，难道也要在孤独中走吗？"
}

for emo, t in texts.items():
    phones, counts = frontend.phones(t)
    print(f"{emo}: {len(phones)} phones, {len(counts)} chars with phones")
