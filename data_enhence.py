from synonyms import synonyms  # pip install synonyms

def synonym_replacement(text, n=1):
    """同义词替换：替换文本中的n个词"""
    words = jieba.lcut(text)
    new_words = words.copy()
    
    random_word_list = list(set([word for word in words if word not in stopwords]))
    random.shuffle(random_word_list)
    
    num_replaced = 0
    for random_word in random_word_list:
        synonyms_list = synonyms.nearby(random_word)[0]
        if len(synonyms_list) > 1:  # 确保有同义词
            synonym = synonyms_list[1]  # 选择最相似的同义词
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n: 
            break
    
    return ' '.join(new_words)

# 通过汉译英，英译汉来增强数据集含义
from googletrans import Translator
def back_translation(text, src_lang='zh-cn', intermediate_lang='en'):
    """回译：中->英->中"""
    try:
        translator = Translator()  # 1. 初始化翻译器
        # 2. 中译英
        en_text = translator.translate(text, src=src_lang, dest=intermediate_lang).text
        # 3. 英译中
        back_text = translator.translate(en_text, src=intermediate_lang, dest=src_lang).text
        return back_text  # 4. 返回回译后的文本
    except:
        return text  # 5. 翻译失败时返回原文本