import spacy
import re

s1 = 'rontal view of chest shows:  Borderline cardiomegaly  .Tortuosity of thoracic aorta  .Hyperinflation and emphysematous 3.2.'
print(s1)
s1 = re.sub(r'(\s{1}[.])', '. ', s1)
print(s1)
s1x = re.sub(r'(\s{1}[.]\s{1})', '. ', s1)
print(s1x)
