from textblob import TextBlob, Word
from textblob.wordnet import VERB, NOUN, Synset


# to install a resource/tool
# import nltk
# nltk.download('wordnet')
# create a textblob
wiki = TextBlob("Python is a high-level, general-purpose programming language.")
print(wiki)
# part-of-speech tagging
print(wiki.tags)
# noun phrase extraction
print(wiki.noun_phrases)
# sentiment analysis 1
testimonial = TextBlob("Textblob is amazingly simple to use. What great fun!")
print(testimonial.sentiment)
# tokenization
zen = TextBlob("Beautiful is better than ugly. Explicit is better than implicit. Simple is better than complex.")
print(zen.words)
# sentence splitter
print(zen.sentences)
# sentiment analysis 2
for sentence in zen.sentences:
    print(sentence.sentiment)
# word inflection
sentence = TextBlob('Use 4 spaces per indentation level.')
print(sentence.words)
print(sentence.words[2].singularize())  # to singularize a word
print(sentence.words[-1].pluralize())  # to pluralize a word
# lemmatization
w = Word("octopi")
print(w.lemmatize())
w = Word("went")
print(w.lemmatize("v"))  # pass in WordNet part of speech (verb)
# wordnet integration
word = Word("octopus")
print(word.synsets)  # to access the synsets
print(Word("hack").get_synsets(pos=VERB))  # to access the synsets
Word("octopus").definitions  # to access the definitions
Word("dog").define(pos=NOUN)  # to access the definitions
# measure the path similarity
octopus = Synset('octopus.n.02')
shrimp = Synset('shrimp.n.03')
print(octopus.path_similarity(shrimp))
# spelling correction
b = TextBlob("I havv goood speling!")
print(b.correct())
w = Word('falibility')
print(w.spellcheck())  # returns correct word with a confidence score
# get word and noun phrase frequencies
monty = TextBlob("We are no longer the Knights who say Ni. We are now the Knights who say Ekki ekki ekki PTANG.")
print(monty.word_counts['ekki'])  # through the word_counts dictionary
print(monty.words.count('ekki'))  # using the count() method
print(monty.words.count('ekki', case_sensitive=True))  # specify case sensitivity
print(wiki.noun_phrases.count('python'))
# translation and language detection
# en_blob = TextBlob(u'Simple is better than complex.')
# print(en_blob.translate(to='es'))
# chinese_blob = TextBlob(u"美丽优于丑陋")
# print(chinese_blob.translate(from_lang="zh-CN", to='en'))
# b = TextBlob(u"بسيط هو أفضل من مجمع")
# print(b.detect_language())
# parsing
b = TextBlob("And now for something completely different.")
print(b.parse())
# textblobs are like python strings!
print(zen[0:19])
print(zen.upper())
print(zen.find("Simple"))
apple_blob = TextBlob('apples')
banana_blob = TextBlob('bananas')
print(apple_blob < banana_blob)
print(apple_blob == 'apples')
apple_blob + ' and ' + banana_blob
TextBlob("apples and bananas")
print("{0} and {1}".format(apple_blob, banana_blob))
# n-grams
blob = TextBlob("Now is better than never.")
print(blob.ngrams(n=3))
# getting start and end indices of sentences
for s in zen.sentences:
    print(s)
    print("---- Starts at index {}, Ends at index {}".format(s.start, s.end))
