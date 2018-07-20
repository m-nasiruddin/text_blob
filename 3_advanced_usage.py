from textblob import TextBlob, Blobber
from textblob.sentiments import NaiveBayesAnalyzer
from nltk.tokenize import TabTokenizer, BlanklineTokenizer
from textblob.np_extractors import ConllExtractor
from textblob.taggers import NLTKTagger
from textblob.parsers import PatternParser


# overriding models and the blobber class
# to install a resource/tool
# import nltk
# nltk.download('movie_reviews')
# nltk.download('conll2000')
# sentiment analyzers
blob = TextBlob("I love this library", analyzer=NaiveBayesAnalyzer())
print(blob.sentiment)
# tokenizers 1
tokenizer = TabTokenizer()
blob = TextBlob("This is\ta rather tabby\tblob.", tokenizer=tokenizer)
print(blob.tokens)
# tokenizers 1
tokenizer = BlanklineTokenizer()
blob = TextBlob("A token\n\nof appreciation")
print(blob.tokenize(tokenizer))
# noun phrase chunkers
extractor = ConllExtractor()
blob = TextBlob("Python is a high-level programming language.", np_extractor=extractor)
print(blob.noun_phrases)
# pos taggers
nltk_tagger = NLTKTagger()
blob = TextBlob("Tag! You're It!", pos_tagger=nltk_tagger)
print(blob.pos_tags)
# parsers
blob = TextBlob("Parsing is fun.", parser=PatternParser())
print(blob.parse())
# blobber: a text blob factory
tb = Blobber(pos_tagger=NLTKTagger())
blob1 = tb("This is a blob.")
blob2 = tb("This is another blob.")
print(blob1.pos_tagger is blob2.pos_tagger)
