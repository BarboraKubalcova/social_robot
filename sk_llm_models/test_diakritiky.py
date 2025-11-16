import ollama, re

original_text = '''
V jednom rybníku žila žaba, ktorá poznala len svoj rybník a zopár mlák. Ale vedela znamenito rozprávať,
kde všade bola, čo všetko videla a čo všetko vie. Rozprávala to tak presvedčivo, že tí, ktorí ju nepoznali, ľahko uverili, že hovorí pravdu.
Toto žabie rozprávanie počula aj myš. Poznala len svoju myšiu dieru a malé políčko v okolí, kam chodila na zrno. Žabie rozprávanie ju očarilo.
Myš bola zvedavá na svet okolo a veľmi by si priala zažiť tiež nejaké dobrodružstvo. „Najlepšie svet spoznáš, moja drahá,“ rozprávala žaba, 
„keď sa naučíš plávať. Po vode sa dostaneš všade. Ver mi, veľmi dobre to poznám. Plávam tak rýchlo, že predbehnem každý parník, a doplávam až k moru, keď sa mi zachce.“
„Ako by som sa ale naučila plávať?“ premýšľala myš.
„Ale prosím ťa, ja ťa to veľmi rada naučím. Naučila som plávať všetky žaby a ryby v tomto rybníku, keď boli ešte žubrienky,“ chvastala sa žaba a ďalej rozprávala, ako to urobia.
„Pozri,“ povedala žaba. „Tu je povrázok. Ja si ťa k sebe priviažem a poplávaš za mnou.“
A tak sa myš priviazala k žabe. Žaba skočila do vody, veselo sa potápala a plávala. Zrazu uvidela ďalšiu žabu.
„Ahoj, sestra! Ako sa máš? Už som ti rozprávala o tom, ako som vyhrala súťaž v potápaní? To bolo tak…“
Žaba sa dala do rozprávania a na myš úplne zabudla. Vôbec si nevšimla, že sa tá chuderka utopila. A žaba si ďalej kvákala s každým, koho stretla.
Uplynuli tri dni. Utopená myš sa nafúkla a vyplávala na hladinu rybníka. Ako sa tak vznášala, zahliadla ju z oblohy kania. Myšku si ulovila k jedlu.
A pretože bola stále priviazaná k žabe, ulovila s myškou aj žabu. A tak sa kania poriadne naobedovala. Zjedla myš aj žabu, pretože mala už veľký hlad.
Na nejaké žabie výmysly ukvákanej žaby nebola vôbec zvedavá.
Žabe to samozrejme patrilo, pretože svojím hlúpym a bezstarostným správaním priviedla do nebezpečenstva myš. Tým privolala nebezpečenstvo
aj sama na seba. Preto sa k sebe správajme ohľaduplne a dávajme pozor jeden na druhého.
'''

corrupted_text = '''
V jednom rybniku zila zaba, ktora poznala len svoj rybnik a zopar mlak. Ale vedela znamenito rozpravat,
kde vsetko bola, co vsetko videla a co vsetko vie. Rozpravala to tak presvedcivo, ze ti, ktori ju nepoznali, lahko uverili, ze hovori pravdu.
Toto zabie rozpravanie pocula aj mys. Poznala len svoju mysiu dieru a male policko v okoli, kam chodila na zrno. Zabie rozpravanie ju ocarilo.
Mys bola zvedava na svet okolo a velmi by si priala zazit tiez nejake dobrodruzstvo. „Najlepsie svet spoznas, moja draha,“ rozpravala zaba,
„ked sa naucis plavat. Po vode sa dostanes vsade. Ver mi, velmi dobre to poznam. Plavam tak rychlo, ze predbehnem kazdy parnik, a doplavam az k moru, ked sa mi zachce.“
„Ako by som sa ale naucila plavat?“ premyslala mys.
„Ale prosim ta, ja ta to velmi rada naucim. Naucila som plavat vsetky zaby a ryby v tomto rybniku, ked boli este zubrienky,“ chvastala sa zaba a dalej rozpravala, ako to urobia.
„Pozri,“ povedala zaba. „Tu je povrazok. Ja si ta k sebe priviazem a poplavas za mnou.“
A tak sa mys priviazala k zabe. Zaba skocila do vody, veselo sa potapala a plavala. Zrazu uvidela dalsiu zaba.
„Ahoj, sestra! Ako sa mas? Uz som ti rozpravala o tom, ako som vyhrala sutaz v potapani? To bolo tak…“
Zaba sa dala do rozpravania a na mys uplne zabudla. Vobec si nevsimla, ze sa ta chuderka utopila. A zaba si dalej kava s kazdym, koho stretla.
Uplynuli tri dni. Utopena mys sa nafukla a vyplavala na hladinu rybnika. Ako sa tak vznasala, zahliadla ju z oblohy kania. Mysku si ulovila k jedlu.
A pretoze bola stale priviazana k zabe, ulovila s myskou aj zaba. A tak sa kania poriadne naobedovala. Zjedla mys aj zaba, pretoze mala uz velky hlad.
Na nejake zabie vymysly ukvakanej zaby nebola vobec zvedava.
Zabe to samozrejme patrilo, pretoze svojim hlupym a bezstarostnym spravanim priviedla do nebezpecenstva mys. Tym privolala nebezpecenstvo
aj sama na seba. Preto sa k sebe spravajme ohladuplne a davajme pozor jeden na druheho.
'''

text_from_gpt = '''
V jednom rybníku žila žaba, ktorá poznala len svoj rybník a zopár mlák. Ale vedela znamenito rozprávať,
kde všade bola, čo všetko videla a čo všetko vie. Rozprávala to tak presvedčivo, že tí, ktorí ju nepoznali, ľahko uverili, že hovorí pravdu.
Toto žabie rozprávanie počula aj myš. Poznala len svoju myšiu dieru a malé políčko v okolí, kam chodila na zrno. Žabie rozprávanie ju očarilo.
Myš bola zvedavá na svet okolo a veľmi by si priala zažiť tiež nejaké dobrodružstvo. „Najlepšie svet spoznáš, moja drahá,“ rozprávala žaba,
„keď sa naučíš plávať. Po vode sa dostaneš všade. Ver mi, veľmi dobre to poznám. Plávam tak rýchlo, že predbehnem každý parník a doplávam až k moru, keď sa mi zachce.“
„Ako by som sa ale naučila plávať?“ premýšľala myš.
„Ale prosím ťa, ja ťa to veľmi rada naučím. Naučila som plávať všetky žaby a ryby v tomto rybníku, keď boli ešte žubrienky,“ chvastala sa žaba a ďalej rozprávala, ako to urobia.
„Pozri,“ povedala žaba. „Tu je povrázok. Ja si ťa k sebe priviažem a poplávaš za mnou.“
A tak sa myš priviazala k žabe. Žaba skočila do vody, veselo sa potápala a plávala. Zrazu uvidela ďalšiu žabu.
„Ahoj, sestra! Ako sa máš? Už som ti rozprávala o tom, ako som vyhrala súťaž v potápaní? To bolo tak…“
Žaba sa dala do rozprávania a na myš úplne zabudla. Vôbec si nevšimla, že sa tá chuderka utopila. A žaba si ďalej kvákala s každým, koho stretla.
Uplynuli tri dni. Utopená myš sa nafúkla a vyplávala na hladinu rybníka. Ako sa tak vznášala, zahliadla ju z oblohy kaňa. Myšku si ulovila k jedlu.
A pretože bola stále priviazaná k žabe, ulovila s myškou aj žabu. A tak sa kaňa poriadne naobedovala. Zjedla myš aj žabu, pretože mala už veľký hlad.
Na nejaké žabie výmysly ukvákanej žaby nebola vôbec zvedavá.
Žabe to samozrejme patrilo, pretože svojím hlúpym a bezstarostným správaním priviedla do nebezpečenstva myš. Tým privolala nebezpečenstvo aj
sama na seba. Preto sa k sebe správajme ohľaduplne a dávajme pozor jeden na druhého.
'''

def use_mistral(prompt):
    model_name = "mistral" 
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.get("message", {}).get("content", "")

prompt_format = '''
    Correct the diacritics in this Slovak text. Return only the corrected text. \n
    {corrupted_text}
'''

prompt = prompt_format.format(
        corrupted_text=corrupted_text
    )

text_from_mistral = use_mistral(prompt)
print(f"Text from mistral:\n {text_from_mistral}")

def clean_text(text):
    # Remove punctuation (dots, commas, etc.) and convert to lowercase
    return re.sub(r'[.,"“”]', '', text).lower()

def calculate_similarity(text1, text2):
    # Clean texts
    words1 = clean_text(text1).split()
    words2 = clean_text(text2).split()
    
    # Find the number of matching words
    matches = sum(1 for w1, w2 in zip(words1, words2) if w1 == w2)
    
    # Calculate the percentage similarity
    total_words = max(len(words1), len(words2))
    similarity = (matches / total_words) * 100 if total_words > 0 else 0
    
    return round(similarity, 2)

sim_corrupted = calculate_similarity(original_text, corrupted_text)
sim_gpt = calculate_similarity(original_text, text_from_gpt)
sim_mistral = calculate_similarity(original_text, text_from_mistral)

print("\n\n")
print(f"Similarity between original and corrupted text: {sim_corrupted}%")
print(f"Similarity between original and text from GPT: {sim_gpt}%")
print(f"Similarity between original and text from mistral: {sim_mistral}%")