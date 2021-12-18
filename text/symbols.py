""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. '''
# from text import cmudict
#
# _pad        = '_'
# _punctuation = '!\'(),.:;? '
# _special = '-'
# _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
#
# # Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
# _arpabet = ['@' + s for s in cmudict.valid_symbols]
#
# # Export all symbols:
# symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) #+ _arpabet


#####################
_pad        = '_'
_eos        = '~'
label = 'phone'    # pinyin  phone
if(label == 'phone'):
    _characters = [
                'breath','cough','noise','smack','um','sil','sp1',
                'a1','a2','a3','a4','a5','ai1','ai2','ai3','ai4','ai5',
                'an1','an2','an3','an4','an5','ang1','ang2','ang3','ang4','ang5',
                'ao1','ao2','ao3','ao4','ao5','b','c','ch','d','e1','e2','e3','e4','e5',
                'ei1','ei2','ei3','ei4','ei5','en1','en2','en3','en4','en5','eng1','eng2',
                'eng3','eng4','eng5','er1','er2','er3','er4','er5','f','g','h','i1','i2',
                'i3','i4','i5','ia1','ia2','ia3','ia4','ia5','ian1','ian2','ian3','ian4','ian5',
                'iang1','iang2','iang3','iang4','iang5','iao1','iao2','iao3','iao4','iao5',
                'ie1','ie2','ie3','ie4','ie5','ii1','ii2','ii3','ii4','ii5','iii1','iii2',
                'iii3','iii4','iii5','in1','in2','in3','in4','in5','ing1','ing2','ing3','ing4',
                'ing5','iong1','iong2','iong3','iong4','iong5','iou1','iou2','iou3','iou4','iou5',
                'j','k','l','m','n','o1','o2','o3','o4','o5','ong1','ong2','ong3','ong4','ong5',
                'ou1','ou2','ou3','ou4','ou5','p','q','r','rr','s','sh','t','u1','u2','u3','u4','u5',
                'ua1','ua2','ua3','ua4','ua5','uai1','uai2','uai3','uai4','uai5','uan1','uan2','uan3',
                'uan4','uan5','uang1','uang2','uang3','uang4','uang5','uei1','uei2','uei3','uei4','uei5',
                'uen1','uen2','uen3','uen4','uen5','uo1','uo2','uo3','uo4','uo5','v1','v2','v3','v4','v5',
                'van1','van2','van3','van4','van5','ve1','ve2','ve3','ve4','ve5','vn1','vn2','vn3','vn4',
                'vn5','w','x','y','z','zh',
                '，',',','！','!','。','.','？','?','、','：',':','；',';',
                '#1','#2','#3','#4','#',' '
                ]
    symbols = [_pad, _eos] + _characters
