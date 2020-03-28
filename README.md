# Tacotron-2(tensorflow) + wavernn(pytorch) 中文语音合成:
  
  
python37，使用标贝数据集，t2模型支持拼音和音素+韵律标签训练(数据量少的情况下，音素+韵律效果比直接拼音好)，修改./tacotron2_wavernn/tacotron/utils/symbols.py和text.py，标签如下：
  
拼音：  
	000001,ka2 er2 pu3 pei2 wai4 sun1 wan2 hua2 ti1  
	000002,jia2 yu3 cun1 yan2 bie2 zai4 yong1 bao4 wo3  
	000003,bao2 ma3 pei4 gua4 bo3 luo2 an1 diao1 chan2 yuan4 zhen3 dong3 weng1 ta4  
  
音素+韵律(dictionary.txt)：  
	000001,k a2 er2 p u3 #2 p ei2 uai4 s uen1 #1 uan2 h ua2 t i1 #4  。   
	000002,j ia2 v3 c uen1 ian2 #2 b ie2 z ai4 #1 iong1 b ao4 uo3 #4  。   
	000003,b ao2 m a3 #1 p ei4 g ua4 #1 b o3 l uo2 an1 #3  ， d iao1 ch an2 #1 van4 zh en3 #2 d ong3 ueng1 t a4 #4  。   
  
  
# Training and Inference:  
  
- Step **(0)**: python preprocess.py ，将原始音频处理成t2训练需要的数据.
- Step **(1)**: python train_tacotron.py ，训练完成后生成gta.
- Step **(2)**: python train_wavernn.py ，用生成的gta训练wavernn.
- Step **(3)**: python inference_tacotron2_wavernn.py ，t2和wavernn训练完成后，可以进行合成测试.
  
  
# Pretrained model and Samples:  
解压后将logs-Tacotron-2拷贝到./tacotron2_wavernn下，将checkpoints拷贝到./tacotron2_wavernn/wavernn_vocoder下  
pre_model(链接: https://pan.baidu.com/s/1EdsFsFF1J4jo4NR9GvAKDA 提取码: u4t5 )  
  
  
# reference:  
https://github.com/Rayhane-mamah/Tacotron-2  
https://github.com/fatchord/WaveRNN  
  
