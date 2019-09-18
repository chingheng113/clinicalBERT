import spacy
import re

s1 = '          1.Peripheral artery occlusive disease2.Syncope, favor arrhythmia-related3.Anemia, favor chronic illness related4.Hypertension5.Benign prostatic hypertrophy 6.Senile cataract'
s2 = '          1. Left side ishemic stroke, with motor aphasia2. diabetes mellitus (3/24 HbA1C= 10.2 %)3. hyperlipidemia'

regex = re.compile('(\d+[.{1}][\D])')
regex2 = re.compile('\n\1')
s1x = re.sub(regex, r'. \1', s1)
s2x = re.sub(regex, r'. \1', s2)
print(s1x)
print(s2x)