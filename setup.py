from distutils.core import setup

setup(
    name='semantic-eventlog-anaylsis'
         '',
    version='1.0',
    packages=['data', 'readwrite', 'preprocessing', 'downstream', 'downstream.instancediscovery', 'downstream.augmentation',
              'evaluation', 'typeandactionextraction', 'typeandactionextraction.bert_tagger', 'typeandactionextraction.labelparser',
              'attributeclassification'],
    url='',
    license='MIT',
    author='Adrian Rebmann',
    author_email='rebmann@informatik.uni-mannheim.de',
    description=''
)
