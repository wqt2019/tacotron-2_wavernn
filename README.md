# tacotron-2(tensorflow) + wavernn(pytorch) chinese TTS:  
  
  
python37，biaobei chinese dataset，tacotron2 support chinese pinyin or chinese phone + rhythm training(default is phone + rhythm)，edit [symbols.py](./tacotron/utils/symbols.py) and [text.py](./tacotron/utils/text.py)：
  
pinyin：  
	000001,ka2 er2 pu3 pei2 wai4 sun1 wan2 hua2 ti1  
	000002,jia2 yu3 cun1 yan2 bie2 zai4 yong1 bao4 wo3  
	000003,bao2 ma3 pei4 gua4 bo3 luo2 an1 diao1 chan2 yuan4 zhen3 dong3 weng1 ta4  
  
phone + rhythm(dictionary.txt)：  
	000001,k a2 er2 p u3 #2 p ei2 uai4 s uen1 #1 uan2 h ua2 t i1 #4  。   
	000002,j ia2 v3 c uen1 ian2 #2 b ie2 z ai4 #1 iong1 b ao4 uo3 #4  。   
	000003,b ao2 m a3 #1 p ei4 g ua4 #1 b o3 l uo2 an1 #3  ， d iao1 ch an2 #1 van4 zh en3 #2 d ong3 ueng1 t a4 #4  。   
  
  
# Training and Inference:  
  
gta:  
- Step **(0)**: python preprocess.py ，process the audios for t2 and wavernn training .  
- Step **(1)**: python train_tacotron.py ，while finish the t2 training, it will generate gta data .  
- Step **(2)**: python train_wavernn.py ，training the wavernn with the gta data .  
- Step **(3)**: python inference_tacotron2_wavernn.py ，while finished the training , you can test the t2 + wavernn tts system.  
  
real mel:  
- Step **(0)**: python preprocess.py ，process the audios for t2 and wavernn training .  
- Step **(1)**: edit [hparams.py](./wavernn_vocoder/hparams.py) , change the 'data_path' that 'gta_path' correspond to the 'mels' dir , and 'audio_path' correspond to the 'audio' dir which are generated in step 0 (line 89 and 90 in ./wavernn_vocoder/utils/dataset.py).  
- Step **(2)**: python train_tacotron.py .  
- Step **(3)**: python train_wavernn.py ，training with the real mel data. 
- Step **(4)**: python inference_tacotron2_wavernn.py ，while finished the training , you can test the t2 + wavernn tts system.  
  
also ,run inference_wavernn.py if you only interested in vocoder .
  
  
# Pretrained model and Samples:  
  
copy the 'logs-Tacotron-2_phone' dir to ./tacotron2_wavernn，copy the 'checkpoints' to ./tacotron2_wavernn/wavernn_vocoder:  
  
pre_model(链接: https://pan.baidu.com/s/1cXMiDoeERomGThig3c4JpA 提取码: qn6d)  
  
  
# reference:  
https://github.com/Rayhane-mamah/Tacotron-2  
https://github.com/fatchord/WaveRNN  
  
