'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run
through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details.
'''
from . import cmudict

_pad        = '_'
_eos        = '~'

label = 'phone'    # pingyin  phone

if(label == 'pinyin'):
    #_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'\"(),-.:;? '
    _characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890@!\'(),-.:;? '

    # Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
    #_arpabet = ['@' + s for s in cmudict.valid_symbols]
    # Export all symbols:
    symbols = [_pad, _eos] + list(_characters) #+ _arpabet


if(label == 'phone'):
    _characters = [
            'breath','cough','noise','smack','um','sil','sp1',
            'a5','a1','a4','ai5','ai1','ai2','ai3','ai4','an1',
            'an2','an3','an4','ang1','ang2','ang4','ao1','ao2',
            'ao3','ao4','b','a2','a3','ang5','ang3','ao5','ei5',
            'ei1','ei2','ei3','ei4','en5','en1','en2','en3','en4',
            'eng1','eng2','eng4','i1','i2','i3','i4','ian5','ian1',
            'ian2','ian3','ian4','iao1','iao2','iao3','iao4','ie1',
            'ie2','ie3','ie4','in1','in4','ing1','ing2','ing3','ing4',
            'o5','o1','o2','o3','o4','u5','u2','u3','u4','c','e4',
            'ch','an5','e1','e2','e3','eng5','eng3','iii5','iii1',
            'iii2','iii3','iii4','ong1','ong2','ong3','ong4','ou5',
            'ou1','ou2','ou3','ou4','u1','uai1','uai3','uai4','uan5',
            'uan1','uan2','uan3','uan4','uang5','uang1','uang2','uang3',
            'uang4','uei5','uei1','uei2','uen1','uen2','uen3','uo1',
            'uo4','ii1','ii2','ii3','ii4','ong5','uei3','uei4','uen4',
            'uo2','d','e5','i5','ia2','ia3','iao5','ie5','iou1','uo5',
            'uo3','eer2','er5','er2','er3','er4','f','g','ua5','ua1',
            'ua2','ua3','ua4','uai2','h','uen5','j','ia5','ia1','ia4',
            'iang5','iang1','iang2','iang3','iang4','in5','in2','in3',
            'ing5','iong2','iong3','iou5','iou2','iou3','iou4','v5',
            'v1','v2','v3','v4','van1','van2','van3','van4','ve1',
            've2','ve4','vn5','vn1','vn4','k','uai5','l','m','n',
            'ng1','p','q','van5','vn2','r','s','sh','ii5','t','ueng1',
            'ueng2','ueng3','ueng4','x','iong5','iong1','ve3','io5',
            'io1','iong4','ve5','vn3','z','zh','，','！','。','？',
            '、','：','#1','#2','#3','#4','#',' '
          ]
    symbols = [_pad, _eos] + _characters
