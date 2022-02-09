import vlc
import time
Instance = vlc.Instance()
player = Instance.media_player_new()
Media = Instance.media_new('D:/SEM 6/PROJECTS/Gesture-Orientation/media/Vaathi Coming.mp4')
player.set_media(Media)
player.play()
time.sleep(20)
player.pause()

