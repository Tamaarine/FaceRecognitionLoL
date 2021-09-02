from selenium import webdriver
import requests
import os
import time

browser = webdriver.Firefox()

browser.get('https://google.com')

# def downloadImages(query):

def fetch_img_urls(query:str, wd:webdriver, sleeptime, image_limit):
    
    search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

    browser.get(search_url.format(q=f"lol {query}"))
    
    browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(sleeptime)
    
    thumbnail_results = wd.find_elements_by_css_selector('img.Q4LuWd')
    
    image_urls = []
    
    post_fix_count = 0
    
    for index, actual_image in enumerate(thumbnail_results):
        actual_image.click()
        
        image_links = wd.find_elements_by_css_selector('img.n3VNCb')
        
        for image_link in image_links:
            if image_link.get_attribute('src') and 'http' in image_link.get_attribute('src') and not 'encrypted' in image_link.get_attribute('src'):
                url = image_link.get_attribute('src')
                try:
                    res = requests.get(url)
                    folder_path = os.path.join(os.getcwd(), 'league', query)
            
                    if not os.path.exists(folder_path):
                        os.mkdir(folder_path)
                    
                    # File path that we are going to store it on
                    file_path = os.path.join(folder_path, str(post_fix_count) + ".jpg")
                    
                    post_fix_count += 1
                    
                    with open(file_path, 'wb') as f:
                        print(f"Writing {file_path}")
                        f.write(res.content)
                except:
                    print("Bad link")
    print(f"Collected {post_fix_count} images from links for {query}")
        
    
    
raw = [
        {
          "value": {
            "id": 69,
            "name": "Cassiopeia",
            "blurb": "Cassiopeia is a terrifying creature - half woman, half snake - whose slightest glance brings death. The youngest daughter of one of Noxus' most influential families, she was once a beautiful and cunning temptress capable of manipulating the hardest...",
            "tags": [
              "Mage"
            ]
          }
        },
        {
          "value": {
            "id": 101,
            "name": "Xerath",
            "blurb": "''A lifetime as a slave has prepared me for eternity as your master.''<br><br>Xerath is an Ascended Magus of ancient Shurima, a being of arcane energy writhing in the broken shards of a magical sarcophagus. For millennia, he was trapped beneath the...",
            "tags": [
              "Mage",
              "Assassin"
            ]
          }
        },
        {
          "value": {
            "id": 115,
            "name": "Ziggs",
            "blurb": "Ziggs was born with a talent for tinkering, but his chaotic, hyperactive nature was unusual among yordle scientists. Aspiring to be a revered inventor like Heimerdinger, he rattled through ambitious projects with manic zeal, emboldened by both his...",
            "tags": [
              "Mage"
            ]
          }
        },
        {
          "value": {
            "id": 5,
            "name": "Xin Zhao",
            "blurb": "''Death is inevitable, one can only avoid defeat.''<br><br>Whenever Jarvan III, the king of Demacia, delivers one of his rallying speeches from the glinting marble balcony atop the Royal Palace, Xin Zhao is at his side. Coined the Seneschal of Demacia...",
            "tags": [
              "Fighter",
              "Assassin"
            ]
          }
        },
        {
          "value": {
            "id": 120,
            "name": "Hecarim",
            "blurb": "''Break their ranks and ride them down without mercy. Crush the living and feast on their terror.''<br><br>Hecarim is an armored colossus who charges from the Shadow Isles at the head of a deathly host of spectral horsemen to hunt the living. A...",
            "tags": [
              "Fighter",
              "Tank"
            ]
          }
        },
        {
          "value": {
            "id": 202,
            "name": "Jhin",
            "blurb": "''Art requires a certain...cruelty.''<br><br>Jhin is a meticulous criminal psychopath who believes murder is art. Once an Ionian prisoner, but freed by shadowy elements within Ionia's ruling council, the serial killer now works as their cabal's assassin...",
            "tags": [
              "Marksman",
              "Assassin"
            ]
          }
        },
        {
          "value": {
            "id": 10,
            "name": "Kayle",
            "blurb": "In a world far away where an ancient war still rages, Kayle was a great hero - the strongest of an immortal race committed to destroying evil wherever it could be found. For ten thousand years, Kayle fought tirelessly for her people, wielding her...",
            "tags": [
              "Fighter",
              "Support"
            ]
          }
        },
        {
          "value": {
            "id": 56,
            "name": "Nocturne",
            "blurb": "Before Nocturne, people believed that dreams were figments of their imagination, meaningless images that flashed through the mind when one slept. This belief was put to the test when a rash of sleep-related incidents started afflicting summoners of the...",
            "tags": [
              "Assassin",
              "Fighter"
            ]
          }
        },
        {
          "value": {
            "id": 42,
            "name": "Corki",
            "blurb": "When Heimerdinger and his yordle colleagues migrated to Piltover, they embraced science as a way of life, and they immediately made several groundbreaking contributions to the techmaturgical community. What yordles lack in stature, they make up for with...",
            "tags": [
              "Marksman"
            ]
          }
        },
        {
          "value": {
            "id": 24,
            "name": "Jax",
            "blurb": "It is seldom the case where a champion is defined by his actions after joining the League of Legends rather than before. Such is the case with Jax, for whom the argument could be made that he is the most prolific tournament fighter currently at the...",
            "tags": [
              "Fighter",
              "Assassin"
            ]
          }
        },
        {
          "value": {
            "id": 37,
            "name": "Sona",
            "blurb": "Sona has no memories of her true parents. As an infant, she was found abandoned on the doorstep of an Ionian adoption house, nestled atop an ancient instrument in an exquisite case of unknown origins. She was an unusually well-behaved child, always...",
            "tags": [
              "Support",
              "Mage"
            ]
          }
        },
        {
          "value": {
            "id": 25,
            "name": "Morgana",
            "blurb": "There is a world far away populated by graceful and beautiful winged beings gifted with immortality, where an ancient conflict still rages. Like so many conflicts, this war split families. One side proclaimed themselves as beings of perfect order and...",
            "tags": [
              "Mage",
              "Support"
            ]
          }
        },
        {
          "value": {
            "id": 14,
            "name": "Sion",
            "blurb": "BLOOD.<br><br>SMELL IT.<br><br>WANT. ACHING. NEED!<br><br>CLOSE NOW. THEY COME.<br><br>NO CHAINS? FREE! KILL!<br><br>IN REACH. YES! DIE! DIE!<br><br>Gone. Too quick. No fight. More. I want... more.<br><br>A voice? Unfamiliar. I see him. The Grand...",
            "tags": [
              "Tank",
              "Fighter"
            ]
          }
        },
        {
          "value": {
            "id": 110,
            "name": "Varus",
            "blurb": "''The life of an arrow is fleeting, built of nothing but direction and intent.''<br><br>For his incomparable skill with the bow and his unquestioned sense of honor, Varus was chosen to be the warden of a sacred Ionian temple. The temple was built to...",
            "tags": [
              "Marksman",
              "Mage"
            ]
          }
        },
        {
          "value": {
            "id": 157,
            "name": "Yasuo",
            "blurb": "Yasuo is a man of resolve, an agile swordsman who wields the wind itself to cut down his foes. This once-proud warrior has been disgraced by a false accusation and forced into a desperate fight for survival. With the world turned against him, he will do...",
            "tags": [
              "Fighter",
              "Assassin"
            ]
          }
        },
        {
          "value": {
            "id": 20,
            "name": "Nunu",
            "blurb": "Sometimes bonds of friendship become stronger than even bonds of blood. When those bonds link a fearless boy to a fearsome Yeti, the bond becomes a force to be reckoned with. Given the responsibility of taming a terrifying beast, Nunu forged a...",
            "tags": [
              "Support",
              "Fighter"
            ]
          }
        },
        {
          "value": {
            "id": 421,
            "name": "Rek'Sai",
            "blurb": "The largest and fiercest of her species, Rek'Sai is a merciless predator that tunnels through the earth to ambush and devour her prey. Her insatiable hunger has laid waste to entire regions of the once-great Shuriman empire. Merchants, traders and armed...",
            "tags": [
              "Fighter"
            ]
          }
        },
        {
          "value": {
            "id": 57,
            "name": "Maokai",
            "blurb": "''All around me are empty husks, soulless and unafraid... but I will bring them fear.''<br><br>Maokai is a rageful, towering treant who fights the unnatural horrors of the Shadow Isles. He was twisted into a force of vengeance after a magical cataclysm...",
            "tags": [
              "Tank",
              "Mage"
            ]
          }
        },
        {
          "value": {
            "id": 78,
            "name": "Poppy",
            "blurb": "''I'm no hero. Just a yordle with a hammer.''<br><br>Runeterra has no shortage of valiant champions, but few are as tenacious as Poppy. Bearing a hammer twice the length of her body, this determined yordle has spent untold years searching for the ''Hero...",
            "tags": [
              "Tank",
              "Fighter"
            ]
          }
        },
        {
          "value": {
            "id": 29,
            "name": "Twitch",
            "blurb": "H.I.V.E. Incident Report<br>Code Violation: Industrial Homicide<br>Casefile Status: Unsolved<br>Investigating Agent: Rol, P.<br><br>Team responded to report of suspicious character, criminal activity; proceeded to Sump Works, Sector 90TZ. Sector 90TZ...",
            "tags": [
              "Marksman",
              "Assassin"
            ]
          }
        },
        {
          "value": {
            "id": 164,
            "name": "Camille",
            "blurb": "Weaponized to execute outside the boundaries of the law, Camille Ferros is an elegant and elite operative who ensures the commerce of the Piltover machine with its Zaunite underbelly runs smoothly. Raised among manners and money, she is the Principal...",
            "tags": [
              "Fighter",
              "Tank"
            ]
          }
        },
        {
          "value": {
            "id": 106,
            "name": "Volibear",
            "blurb": "The unforgiving northern reaches of the Freljord are home to the Ursine, a fierce and warlike race that has endured the barren tundra for thousands of years. Their leader is a furious adversary who commands the force of lightning to strike fear within...",
            "tags": [
              "Fighter",
              "Tank"
            ]
          }
        },
        {
          "value": {
            "id": 238,
            "name": "Zed",
            "blurb": "Zed is the first ninja in 200 years to unlock the ancient, forbidden ways. He defied his clan and master, casting off the balance and discipline that had shackled him all his life. Zed now offers power to those who embrace knowledge of the shadows, and...",
            "tags": [
              "Assassin",
              "Fighter"
            ]
          }
        },
        {
          "value": {
            "id": 121,
            "name": "Kha'Zix",
            "blurb": "A vicious Void predator, Kha'Zix infiltrated Valoran to devour the land's most promising creatures. With each kill he absorbs his prey's strength, evolving to grow more powerful. Kha'Zix hungers most to conquer and consume Rengar, the one beast he...",
            "tags": [
              "Assassin",
              "Fighter"
            ]
          }
        },
        {
          "value": {
            "id": 84,
            "name": "Akali",
            "blurb": "There exists an ancient order originating in the Ionian Isles dedicated to the preservation of balance. Order, chaos, light, darkness -- all things must exist in perfect harmony for such is the way of the universe. This order is known as the Kinkou and...",
            "tags": [
              "Assassin"
            ]
          }
        },
        {
          "value": {
            "id": 61,
            "name": "Orianna",
            "blurb": "There once was a Piltovian man named Corin Reveck who had a daughter named Orianna, whom he loved more than anything else in the world. Though Orianna had incredible talent for dancing, she was deeply fascinated by the champions of the League of Legends...",
            "tags": [
              "Mage",
              "Support"
            ]
          }
        },
        {
          "value": {
            "id": 89,
            "name": "Leona",
            "blurb": "''If you would shine like a sun, first you must burn like one.''<br><br>Imbued with the fire of the sun, Leona is a warrior templar of the Solari who defends Mount Targon with her Zenith Blade and Shield of Daybreak. Her skin shimmers with starfire...",
            "tags": [
              "Tank",
              "Support"
            ]
          }
        },
        {
          "value": {
            "id": 133,
            "name": "Quinn",
            "blurb": "Quinn and Valor are an elite ranger team. With crossbow and claw, they undertake their nation's most dangerous missions deep within enemy territory, from swift reconnaissance to lethal strikes. The pair's unbreakable bond is deadly on the battlefield...",
            "tags": [
              "Marksman",
              "Fighter"
            ]
          }
        },
        {
          "value": {
            "id": 1,
            "name": "Annie",
            "blurb": "There have always been those within Noxus who did not agree with the evils perpetrated by the Noxian High Command. The High Command had just put down a coup attempt from the self-proclaimed Crown Prince Raschallion, and a crackdown on any form of...",
            "tags": [
              "Mage"
            ]
          }
        },
        {
          "value": {
            "id": 74,
            "name": "Heimerdinger",
            "blurb": "From the Journal of Professor Cecil B. Heimerdinger<br><br>10.14<br><br>09:15<br><br>Current meteorological conditions in Bandle City seem optimal. Atmospheric pressure is ideal for today's experiments!<br><br>Running a fifth trial for my...",
            "tags": [
              "Mage",
              "Support"
            ]
          }
        },
        {
          "value": {
            "id": 6,
            "name": "Urgot",
            "blurb": "There are warriors who become great for their strength, cunning, or skill with arms. Others simply refuse to die. Urgot, once a great soldier of Noxus, may constitute a case in support of the latter. Prone to diving headlong into enemy battle lines...",
            "tags": [
              "Marksman",
              "Fighter"
            ]
          }
        },
        {
          "value": {
            "id": 60,
            "name": "Elise",
            "blurb": "''Beauty is power too, and can strike swifter than any sword.''<br><br>Elise is a deadly predator who dwells in a shuttered, lightless palace, deep in the Immortal Bastion of Noxus. Once she was mortal, the mistress of a once-powerful house, but the...",
            "tags": [
              "Mage",
              "Fighter"
            ]
          }
        },
        {
          "value": {
            "id": 117,
            "name": "Lulu",
            "blurb": "Perhaps more than any other champion in the League, Lulu marches to the beat of her own drum. During her youth in Bandle City, she spent most of her time wandering alone in the forest or lost in a daydream. It wasn't that she was antisocial; the...",
            "tags": [
              "Support",
              "Mage"
            ]
          }
        },
        {
          "value": {
            "id": 85,
            "name": "Kennen",
            "blurb": "There exists an ancient order originating in the Ionian Isles dedicated to the preservation of balance. Order, chaos, light, darkness -- all things must exist in perfect harmony for such is the way of the universe. This order is known as the Kinkou and...",
            "tags": [
              "Mage",
              "Marksman"
            ]
          }
        },
        {
          "value": {
            "id": 201,
            "name": "Braum",
            "blurb": "''Would you like a bedtime story?''<br><br>''Grandma, I'm too old for that.''<br><br>''You're never too old to be told a story.''<br><br>The girl reluctantly crawls into bed and waits, knowing she won't win this battle. A bitter wind howls outside...",
            "tags": [
              "Support",
              "Tank"
            ]
          }
        },
        {
          "value": {
            "id": 102,
            "name": "Shyvana",
            "blurb": "A half-breed born from the union between dragon and human, Shyvana searched all her life for belonging. Persecution forged her into a brutal warrior, and those who dare stand against Shyvana face the fiery beast lurking just beneath her skin...",
            "tags": [
              "Fighter",
              "Tank"
            ]
          }
        },
        {
          "value": {
            "id": 28,
            "name": "Evelynn",
            "blurb": "Swift and lethal, Evelynn is one of the most deadly - and expensive - assassins in all of Runeterra. Able to merge with the shadows at will, she patiently stalks her prey, waiting for the right moment to strike. While Evelynn is clearly not entirely...",
            "tags": [
              "Assassin",
              "Mage"
            ]
          }
        },
        {
          "value": {
            "id": 38,
            "name": "Kassadin",
            "blurb": "There is a place between dimensions and between worlds. To some it is known as the Outside, to others it is the Unknown. To most, however, it is called the Void. Despite its name, the Void is not an empty place, but rather the home of unspeakable things...",
            "tags": [
              "Assassin",
              "Mage"
            ]
          }
        },
        {
          "value": {
            "id": 429,
            "name": "Kalista",
            "blurb": "''When wronged, we seek justice. When hurt, we strike back. When betrayed, the Spear of Vengeance strikes!''<br><br>A specter of wrath and retribution, Kalista is the undying spirit of vengeance, an armored nightmare summoned from the Shadow Isles to...",
            "tags": [
              "Marksman"
            ]
          }
        },
        {
          "value": {
            "id": 21,
            "name": "Miss Fortune",
            "blurb": "''The bigger the risk, the bigger the bounty.''<br><br>Beauty and danger: There are few who can match Miss Fortune in either. One of Bilgewater's most infamous bounty hunters, she built her legend upon a swathe of bullet-riddled corpses and captured...",
            "tags": [
              "Marksman"
            ]
          }
        },
        {
          "value": {
            "id": 33,
            "name": "Rammus",
            "blurb": "''OK.''<br><br>Idolized by many, dismissed by some, mystifying to all, the curious being, Rammus, is an enigma. Protected by a spiked shell, Rammus inspires increasingly disparate theories on his origin wherever he goes - from demigod, to sacred oracle...",
            "tags": [
              "Tank",
              "Fighter"
            ]
          }
        },
        {
          "value": {
            "id": 92,
            "name": "Riven",
            "blurb": "''There is a place between war and murder in which our demons lurk.''<br><br>In Noxus, any citizen may rise to power regardless of race, gender, or social standing - strength is all that matters. It was with committed faith in this ideal that Riven...",
            "tags": [
              "Fighter",
              "Assassin"
            ]
          }
        },
        {
          "value": {
            "id": 9,
            "name": "Fiddlesticks",
            "blurb": "For nearly twenty years, Fiddlesticks has stood alone in the easternmost summoning chamber of the Institute of War. Only the burning emerald light of his unearthly gaze pierces the musty darkness of his dust-covered home. It is here that the Harbinger...",
            "tags": [
              "Mage",
              "Support"
            ]
          }
        },
        {
          "value": {
            "id": 53,
            "name": "Blitzcrank",
            "blurb": "Zaun is a place where both magic and science have gone awry, and the unchecked nature of experimentation has taken its toll. However, Zaun's lenient restrictions allow their researchers and inventors the leeway to push the bounds of science at an...",
            "tags": [
              "Tank",
              "Fighter"
            ]
          }
        },
        {
          "value": {
            "id": 141,
            "name": "Kayn",
            "blurb": "“The child is gone. The killer remains.”<br><br>A peerless practitioner of lethal shadow magic, Shieda Kayn battles to achieve his true destiny—to one day lead the Order of the Shadow into a new era of Ionian supremacy. He audaciously wields the...",
            "tags": [
              "Fighter",
              "Assassin"
            ]
          }
        },
        {
          "value": {
            "id": 420,
            "name": "Illaoi",
            "blurb": "''I'm not big on sermons. Broken bones teach better lessons.''<br>Illaoi's powerful physique is dwarfed only by her indomitable faith. As the prophet of the Great Kraken, she uses a huge, golden idol to rip her foes' spirits from their bodies and...",
            "tags": [
              "Fighter",
              "Tank"
            ]
          }
        },
        {
          "value": {
            "id": 77,
            "name": "Udyr",
            "blurb": "Udyr is more than a man; he is a vessel for the untamed power of four primal animal spirits. When tapping into the spirits' bestial natures, Udyr can harness their unique strengths: the tiger grants him speed and ferocity, the turtle resilience, the...",
            "tags": [
              "Fighter",
              "Tank"
            ]
          }
        },
        {
          "value": {
            "id": 96,
            "name": "Kog'Maw",
            "blurb": "''If that's just hungry, I don't want to see angry.''<br><br>When the prophet Malzahar was reborn in Icathia, he was led there by an ominous voice which thereafter anchored itself to his psyche. From within, this voice bestowed upon him terrible purpose...",
            "tags": [
              "Marksman",
              "Mage"
            ]
          }
        },
        {
          "value": {
            "id": 13,
            "name": "Ryze",
            "blurb": "''Take care with this world. What is made can be unmade.''<br><br>Widely considered one of the most adept sorcerers on Runeterra, Ryze is an ancient, hard-bitten archmage with an impossibly heavy burden to bear. Armed with a boundless constitution and a...",
            "tags": [
              "Mage",
              "Fighter"
            ]
          }
        },
        {
          "value": {
            "id": 41,
            "name": "Gangplank",
            "blurb": "''I was cutting throats and sinking Noxian war galleys when you were still pissing your britches, boy. You don't want to take me on.''<br><br>As unpredictable as he is brutal, the dethroned reaver king known as Gangplank is feared far and wide. Where he...",
            "tags": [
              "Fighter"
            ]
          }
        },
        {
          "value": {
            "id": 134,
            "name": "Syndra",
            "blurb": "Born with immense magical potential, Syndra loves nothing more than exercising the incredible power at her command. With each passing day, her mastery of magical force grows more potent and devastating. Refusing any notion of balance or restraint...",
            "tags": [
              "Mage",
              "Support"
            ]
          }
        },
        {
          "value": {
            "id": 105,
            "name": "Fizz",
            "blurb": "Centuries ago, an ancient water-dwelling race built a hidden city beneath a mountain in the sea. Though these creatures had their enemies, the city was an impenetrable fortress, and, in the safety it provided, they grew complacent. Fizz, however...",
            "tags": [
              "Assassin",
              "Fighter"
            ]
          }
        },
        {
          "value": {
            "id": 2,
            "name": "Olaf",
            "blurb": "Most men would say that death is a thing to be feared; none of those men would be Olaf. The Berserker lives only for the roar of a battle cry and the clash of steel. Spurred on by his hunger for glory and the looming curse of a forgettable death, Olaf...",
            "tags": [
              "Fighter",
              "Tank"
            ]
          }
        },
        {
          "value": {
            "id": 266,
            "name": "Aatrox",
            "blurb": "Aatrox is a legendary warrior, one of only five that remain of an ancient race known as the Darkin. He wields his massive blade with grace and poise, slicing through legions in a style that is hypnotic to behold. With each foe felled, Aatrox's seemingly...",
            "tags": [
              "Fighter",
              "Tank"
            ]
          }
        },
        {
          "value": {
            "id": 412,
            "name": "Thresh",
            "blurb": "''The mind is a wondrous thing to tear apart.''<br><br>Sadistic and cunning, Thresh is a restless spirit who prides himself on tormenting mortals and breaking them with slow, excruciating inventiveness. His victims suffer far beyond the point of death...",
            "tags": [
              "Support",
              "Fighter"
            ]
          }
        },
        {
          "value": {
            "id": 32,
            "name": "Amumu",
            "blurb": "''Solitude can be lonelier than death.''<br><br>A lonely and melancholy soul from ancient Shurima, Amumu roams the world in search of a friend. Cursed by an ancient spell, he is doomed to remain alone forever, as his touch is death and his affection...",
            "tags": [
              "Tank",
              "Mage"
            ]
          }
        },
        {
          "value": {
            "id": 34,
            "name": "Anivia",
            "blurb": "Anivia is a being of the coldest winter, a mystical embodiment of ice magic, and an ancient protector of the Freljord. She commands all the power and fury of the land itself, calling the snow and bitter wind to defend her home from those who would harm...",
            "tags": [
              "Mage",
              "Support"
            ]
          }
        },
        {
          "value": {
            "id": 45,
            "name": "Veigar",
            "blurb": "To most, thoughts of yordles do not conjure images to be feared. The easygoing half-pint race, though fierce, is often regarded with some degree of joviality. Their high-pitched voices and naturally cute forms inspire something of a protective instinct...",
            "tags": [
              "Mage"
            ]
          }
        },
        {
          "value": {
            "id": 161,
            "name": "Vel'Koz",
            "blurb": "I pass into the sudden glare. Blink. Blink, blink, blink. My eyes adjust and evaluate the landscape before me.<br><br>There's a scurrying. I look down to find a small, white creature standing on its hind legs, sniffing at my body. It intrigues me...",
            "tags": [
              "Mage"
            ]
          }
        },
        {
          "value": {
            "id": 64,
            "name": "Lee Sin",
            "blurb": "As a young teen, Lee Sin was intent on becoming a summoner. His will and dedication were unmatched by any of his peers, and his skill drew the attention of Reginald Ashram, the League's High Councilor at the time. While studying at the Arcanum Majoris...",
            "tags": [
              "Fighter",
              "Assassin"
            ]
          }
        },
        {
          "value": {
            "id": 17,
            "name": "Teemo",
            "blurb": "Teemo is a legend among his yordle brothers and sisters in Bandle City. As far as yordles are concerned, there is something just slightly off about him. While Teemo enjoys the companionship of other yordles, he also insists on frequent solo missions in...",
            "tags": [
              "Marksman",
              "Assassin"
            ]
          }
        },
        {
          "value": {
            "id": 22,
            "name": "Ashe",
            "blurb": "With each arrow she fires from her ancient ice-enchanted bow, Ashe proves she is a master archer. She chooses each target carefully, waits for the right moment, and then strikes with power and precision. It is with this same vision and focus that she...",
            "tags": [
              "Marksman",
              "Support"
            ]
          }
        },
        {
          "value": {
            "id": 44,
            "name": "Taric",
            "blurb": "''The best weapons are beautiful.''<br><br>Taric is the Aspect of the Protector, wielding incredible power as Runeterra's guardian of life, love, and beauty. Shamed by a dereliction of duty and exiled from his homeland Demacia, Taric ascended Mount...",
            "tags": [
              "Support",
              "Fighter"
            ]
          }
        },
        {
          "value": {
            "id": 59,
            "name": "Jarvan IV",
            "blurb": "''There is only one truth, and you will find it at the point of my lance.''<br><br>As the royal family of Demacia for centuries, members of the Lightshield line have spent their lives waging war against any who opposed Demacian ethics. It is said that...",
            "tags": [
              "Tank",
              "Fighter"
            ]
          }
        },
        {
          "value": {
            "id": 27,
            "name": "Singed",
            "blurb": "Singed descended from a long line of Zaun's revered chemists. Even in his youth, his talent for concocting potions far outstripped that of his peers, and he quickly distinguished himself from his less extraordinary chemist compatriots. It came as no...",
            "tags": [
              "Tank",
              "Fighter"
            ]
          }
        },
        {
          "value": {
            "id": 12,
            "name": "Alistar",
            "blurb": "As the mightiest warrior to ever emerge from the Minotaur tribes of the Great Barrier, Alistar defended his tribe from Valoran's many dangers; that is, until the coming of the Noxian army. Alistar was lured from his village by the machinations of Keiran...",
            "tags": [
              "Tank",
              "Support"
            ]
          }
        },
        {
          "value": {
            "id": 54,
            "name": "Malphite",
            "blurb": "There is a world of perfect harmony, where all are part of the whole. The Monolith is the essence of all creation, and its denizens are but singular pieces of it. It is beautiful in its symmetry, and in its almost complete lack of uncertainty. The rocky...",
            "tags": [
              "Tank",
              "Fighter"
            ]
          }
        },
        {
          "value": {
            "id": 498,
            "name": "Xayah",
            "blurb": "Deadly and precise, Xayah is a vastayan revolutionary waging a personal war to save her people. She uses her speed, guile, and razor-sharp feather blades to cut down anyone who stands in her way. Xayah fights alongside her partner and lover, Rakan, to...",
            "tags": [
              "Marksman"
            ]
          }
        },
        {
          "value": {
            "id": 236,
            "name": "Lucian",
            "blurb": "Lucian wields relic weapons imbued with ancient power and stands a stalwart guardian against the undead. His cold conviction never wavers, even in the face of the maddening horrors he destroys beneath his hail of purifying fire. Lucian walks alone on a...",
            "tags": [
              "Marksman"
            ]
          }
        },
        {
          "value": {
            "id": 86,
            "name": "Garen",
            "blurb": "Throughout Valoran, the resolve of Demacia's military is alternately celebrated or despised, but always respected. Their ''zero tolerance'' moral code is strictly upheld by civilians and soldiers alike. In combat, this means Demacian troops may not make...",
            "tags": [
              "Fighter",
              "Tank"
            ]
          }
        },
        {
          "value": {
            "id": 113,
            "name": "Sejuani",
            "blurb": "Sejuani was weaned on hardship and reared on barbarity. Where others succumbed to the harshness of the Freljord, she was tempered by it until pain became power, hunger an encouragement, and frost an ally in culling the weak. Through her ordeals, she...",
            "tags": [
              "Tank",
              "Fighter"
            ]
          }
        },
        {
          "value": {
            "id": 81,
            "name": "Ezreal",
            "blurb": "The intrepid young adventurer Ezreal has explored some of the most remote and abandoned locations on Runeterra. During an expedition to the buried ruins of ancient Shurima, he recovered an amulet of incredible mystical power. Likely constructed to be...",
            "tags": [
              "Marksman",
              "Mage"
            ]
          }
        },
        {
          "value": {
            "id": 76,
            "name": "Nidalee",
            "blurb": "There are few dwellers, let alone champions, residing in the blasted and dangerous lands that lie south of the Great Barrier. Much of that world still bears the scars of past Runes Wars, especially the mysterious Kumungu Jungle. There are long-forgotten...",
            "tags": [
              "Assassin",
              "Fighter"
            ]
          }
        },
        {
          "value": {
            "id": 7,
            "name": "LeBlanc",
            "blurb": "Every city has its dark side, even one whose reputation is already of a questionable hue. Noxus - though its name is already invoked with a mixture of reverence and revulsion - is no exception to this simple truth. Deep within the winding dungeons that...",
            "tags": [
              "Assassin",
              "Mage"
            ]
          }
        },
        {
          "value": {
            "id": 245,
            "name": "Ekko",
            "blurb": "A prodigy from the rough streets of Zaun, Ekko manipulates time to spin any situation to his advantage. Using his own invention, the Zero-Drive, he explores the branching possibilities of reality. As well as experimenting with multi-dimensional...",
            "tags": [
              "Assassin",
              "Fighter"
            ]
          }
        },
        {
          "value": {
            "id": 39,
            "name": "Irelia",
            "blurb": "''The sword flourishes, as though painting with blood.''<br><br>The Ionians have developed some of the most breathtaking and deadly martial arts in all of Runeterra - just one manifestation of their pursuit of enlightenment. The most remarkable blade...",
            "tags": [
              "Fighter",
              "Assassin"
            ]
          }
        },
        {
          "value": {
            "id": 98,
            "name": "Shen",
            "blurb": "''The Eye is blind to fear, to hate, to love - to all things that would sway equilibrium.''<br><br>Leader of a secret clan of mystic warriors, Shen serves as the Eye of Twilight, entrusted to enforce equilibrium in the world. Longing to remain free from...",
            "tags": [
              "Tank,melee"
            ]
          }
        },
        {
          "value": {
            "id": 103,
            "name": "Ahri",
            "blurb": "Unlike other foxes that roamed the woods of southern Ionia, Ahri had always felt a strange connection to the magical world around her; a connection that was somehow incomplete. Deep inside, she felt the skin she had been born into was an ill fit for her...",
            "tags": [
              "Mage",
              "Assassin"
            ]
          }
        },
        {
          "value": {
            "id": 91,
            "name": "Talon",
            "blurb": "''The three deadliest blademasters in all of Valoran are bound to the house of Du Couteau: my father, myself, and Talon. Challenge us, if you dare.''<br>-- Katarina Du Couteau<br><br>Talon's earliest memories are the darkness of Noxus' underground...",
            "tags": [
              "Assassin",
              "Fighter"
            ]
          }
        },
        {
          "value": {
            "id": 240,
            "name": "Kled",
            "blurb": "''A sane man would run . . . but I ain't the runnin' kind!''<br><br>A warrior as fearless as he is ornery, Kled is a popular folk hero in Noxus. Embodying the furious bravado of his nation, he is an icon beloved by the empire's soldiers, distrusted by...",
            "tags": [
              "Fighter",
              "Tank"
            ]
          }
        },
        {
          "value": {
            "id": 223,
            "name": "Tahm Kench",
            "blurb": "''The whole world's a river, and I'm its king.''<br>Tahm Kench travels Runeterra's waterways, feeding his insatiable appetite with the misery of the unsuspecting. The singularly charming gourmand savors every moment of his victims' suffering.  A deal...",
            "tags": [
              "Support",
              "Tank"
            ]
          }
        },
        {
          "value": {
            "id": 3,
            "name": "Galio",
            "blurb": "Outside the gleaming city of Demacia, the stone colossus Galio keeps vigilant watch. Built as a bulwark against enemy mages, he often stands motionless for decades until the presence of powerful magic stirs him to life. Once activated, Galio makes the...",
            "tags": [
              "Tank",
              "Mage"
            ]
          }
        },
        {
          "value": {
            "id": 267,
            "name": "Nami",
            "blurb": "Nami channels the primal energies of the ocean, harnessing its mystical restorative properties and commanding the raw power of the tides themselves. Though many doubted her, Nami had the bravery and determination to take on a dangerous quest when no one...",
            "tags": [
              "Support",
              "Mage"
            ]
          }
        },
        {
          "value": {
            "id": 80,
            "name": "Pantheon",
            "blurb": "''Bring forth one true champion, or a hundred more like you, and then we shall have a battle that will be spoken of until the end of time.''<br><br>The peerless warrior known as Pantheon is a nigh-unstoppable paragon of battle. He was born among the...",
            "tags": [
              "Fighter",
              "Assassin"
            ]
          }
        },
        {
          "value": {
            "id": 35,
            "name": "Shaco",
            "blurb": "Most would say that death isn't funny. It isn't, unless you're Shaco - then it's hysterical. He is Valoran's first fully functioning homicidal comic; he jests until someone dies, and then he laughs. The figure that has come to be known as the Demon...",
            "tags": [
              "Assassin"
            ]
          }
        },
        {
          "value": {
            "id": 112,
            "name": "Viktor",
            "blurb": "Early in life, Viktor discovered his passion for science and invention, particularly in the field of mechanical automation. He attended Zaun's prestigious College of Techmaturgy and led the team that constructed Blitzcrank - a scientific breakthrough...",
            "tags": [
              "Mage"
            ]
          }
        },
        {
          "value": {
            "id": 48,
            "name": "Trundle",
            "blurb": "Trundle is a hulking and devious troll with a mischievous streak. There is nothing he can't beat into submission and bend to his will, not even the ice itself. With his massive, frozen club, he chills his enemies to the core and runs them through with...",
            "tags": [
              "Fighter",
              "Tank"
            ]
          }
        },
        {
          "value": {
            "id": 63,
            "name": "Brand",
            "blurb": "In a faraway place known as Lokfar there was a seafaring marauder called Kegan Rodhe. As was his people's way, Kegan sailed far and wide with his fellows, stealing treasures from those unlucky enough to catch their attention. To some, he was a monster;...",
            "tags": [
              "Mage"
            ]
          }
        },
        {
          "value": {
            "id": 18,
            "name": "Tristana",
            "blurb": "Greatness comes in all shapes and sizes, as proven by this diminutive, cannon-wielding yordle. In a world fraught with turmoil, Tristana refuses to back down from any challenge. She represents the pinnacle of martial proficiency, unwavering courage, and...",
            "tags": [
              "Marksman",
              "Assassin"
            ]
          }
        },
        {
          "value": {
            "id": 150,
            "name": "Gnar",
            "blurb": "The jungle does not forgive blindness. Every broken branch tells a story.<br><br>I've hunted every creature this jungle has to offer. I was certain there were no challenges left here, but now there is something new. Each track is the size of a tusklord;...",
            "tags": [
              "Fighter",
              "Tank"
            ]
          }
        },
        {
          "value": {
            "id": 50,
            "name": "Swain",
            "blurb": "The earliest account of Swain's existence comes from a Noxian infirmary doctor's notes. According to them, Swain limped into the ward without cry or complaint; his right leg was snapped in half, with bone protruding from the skin. A small, scowling bird...",
            "tags": [
              "Mage",
              "Fighter"
            ]
          }
        },
        {
          "value": {
            "id": 67,
            "name": "Vayne",
            "blurb": "The world is not always as civilized as people might think. There are still those who would follow the blackest paths of magic and become corrupted by the darker powers that flow through Runeterra. Shauna Vayne knows this fact well.<br><br>As a young...",
            "tags": [
              "Marksman",
              "Assassin"
            ]
          }
        },
        {
          "value": {
            "id": 16,
            "name": "Soraka",
            "blurb": "A healer gifted with the magic of the stars, Soraka holds all living creatures close to her heart. She was once a celestial being, but she sacrificed her immortality and entered the world of mortals. So long as evil threatens life in Valoran, Soraka...",
            "tags": [
              "Support",
              "Mage"
            ]
          }
        },
        {
          "value": {
            "id": 127,
            "name": "Lissandra",
            "blurb": "Lissandra's magic twists the pure power of ice into something dark and terrible. With the force of her black ice, she does more than freeze - she impales and crushes those who oppose her. To the terrified denizens of the north, she is known only as...",
            "tags": [
              "Mage"
            ]
          }
        },
        {
          "value": {
            "id": 31,
            "name": "Cho'Gath",
            "blurb": "There is a place between dimensions, between worlds. To some it is known as the Outside, to others it is the Unknown. To those that truly know, however, it is called the Void. Despite its name, the Void is not an empty place, but rather the home of...",
            "tags": [
              "Tank",
              "Mage"
            ]
          }
        },
        {
          "value": {
            "id": 154,
            "name": "Zac",
            "blurb": "Zac is the product of a Zaun experiment to manufacture a hexchem-engineered supersoldier - the Zaun Amorphous Combatant. Combining brute strength with limitless flexibility, he is a versatile juggernaut: a creative fighter who bounces over obstacles and...",
            "tags": [
              "Tank",
              "Fighter"
            ]
          }
        },
        {
          "value": {
            "id": 11,
            "name": "Master Yi",
            "blurb": "Through the ancient martial art of Wuju, Master Yi has tempered his body and sharpened his mind until thought and action have become one. Though he chooses to enter into violence as a last resort, the grace and speed with which he wields his blade...",
            "tags": [
              "Assassin",
              "Fighter"
            ]
          }
        },
        {
          "value": {
            "id": 72,
            "name": "Skarner",
            "blurb": "''We are one. We cannot be shattered.''<br><br>Skarner is an immense crystalline scorpion from a hidden valley in Shurima. Part of the ancient Brackern race, Skarner and his kin are known for their great wisdom and deep connection to the land, as their...",
            "tags": [
              "Fighter",
              "Tank"
            ]
          }
        },
        {
          "value": {
            "id": 143,
            "name": "Zyra",
            "blurb": "Longing to take control of her fate, the ancient, dying plant Zyra transferred her consciousness into a human body for a second chance at life. Centuries ago, she and her kind dominated the Kumungu Jungle, using thorns and vines to consume any animal...",
            "tags": [
              "Mage",
              "Support"
            ]
          }
        },
        {
          "value": {
            "id": 497,
            "name": "Rakan",
            "blurb": "As mercurial as he is charming, Rakan is an infamous vastayan troublemaker and the greatest battle-dancer in Lhotlan tribal history. To the humans of the Ionian highlands, his name has long been synonymous with wild festivals, uncontrollable parties...",
            "tags": [
              "Support"
            ]
          }
        },
        {
          "value": {
            "id": 43,
            "name": "Karma",
            "blurb": "Karma is a woman of indomitable will and unbound spiritual power. She is the soul of Ionia made manifest and an inspiring presence on the battlefield, shielding her allies and turning back her foes. A strong leader torn between tradition and revolution...",
            "tags": [
              "Mage",
              "Support"
            ]
          }
        },
        {
          "value": {
            "id": 99,
            "name": "Lux",
            "blurb": "Born to the prestigious Crownguards, the paragon family of Demacian service, Luxanna was destined for greatness. She grew up as the family's only daughter, and she immediately took to the advanced education and lavish parties required of families as...",
            "tags": [
              "Mage",
              "Support"
            ]
          }
        },
        {
          "value": {
            "id": 203,
            "name": "Kindred",
            "blurb": "''Tell me again, little Lamb, which things are ours to take?''<br>''All things, Dear Wolf.''<br>Separate, but never parted, Kindred represents the twin essences of death. Lamb's arrow offers a swift release for those who accept their fate. Wolf hunts...",
            "tags": [
              "Marksman"
            ]
          }
        },
        {
          "value": {
            "id": 104,
            "name": "Graves",
            "blurb": "Malcolm Graves is a wanted man in every realm, city and empire he has visited. Tough, strong-willed, and above all, relentless, through his life of crime he has amassed (then invariably lost) a small fortune.",
            "tags": [
              "Marksman"
            ]
          }
        },
        {
          "value": {
            "id": 40,
            "name": "Janna",
            "blurb": "There are those sorcerers who give themselves over to the primal powers of nature, forgoing the learned practice of magic. Such a sorceress is Janna, who first learned magic as an orphan growing up amidst the chaos that is the city-state of Zaun. Janna...",
            "tags": [
              "Support",
              "Mage"
            ]
          }
        },
        {
          "value": {
            "id": 26,
            "name": "Zilean",
            "blurb": "In the wastelands of Urtistan, there was once a great city. It perished long ago in a terrible Rune War, like most of the lands below the Great Barrier. Nevertheless, one man survived: a sorcerer named Zilean. Being obsessed with time, it was only...",
            "tags": [
              "Support",
              "Mage"
            ]
          }
        },
        {
          "value": {
            "id": 55,
            "name": "Katarina",
            "blurb": "Driven by an intense killer instinct, Katarina uses her talents as an assassin for the glory of Noxus, and the continued elevation of her family. While her fervor drives her to ever-greater feats, it can sometimes lead her astray.<br><br>From childhood...",
            "tags": [
              "Assassin",
              "Mage"
            ]
          }
        },
        {
          "value": {
            "id": 114,
            "name": "Fiora",
            "blurb": "''I have come to kill you for the sake of honor. And though you possess none, still you die.''<br>The most feared duelist in all Valoran, Fiora is as renowned for her brusque manner and cunning mind as she is for the speed of her bluesteel rapier. Born...",
            "tags": [
              "Fighter",
              "Assassin"
            ]
          }
        },
        {
          "value": {
            "id": 23,
            "name": "Tryndamere",
            "blurb": "Fueled by his unbridled fury and rage, Tryndamere cuts his way through the tundra, mastering the art of battle by challenging the Freljord's greatest warriors. The wrathful barbarian seeks revenge on the one who decimated his clan and strikes down all...",
            "tags": [
              "Fighter",
              "Assassin"
            ]
          }
        },
        {
          "value": {
            "id": 8,
            "name": "Vladimir",
            "blurb": "There is a temple hidden in the mountains between Noxus and the Tempest Flats, where the secrets of an ancient and terrifying sorcery are kept. The area surrounding the temple is littered with the exsanguinated corpses of those who have mistakenly...",
            "tags": [
              "Mage",
              "Tank"
            ]
          }
        },
        {
          "value": {
            "id": 75,
            "name": "Nasus",
            "blurb": "''What was fallen will be great again.''<br><br>Nasus is an imposing, jackal-headed Ascended being from ancient Shurima, a heroic figure regarded as a demigod by the people of the desert. Fiercely intelligent, he was a guardian of knowledge and peerless...",
            "tags": [
              "Fighter",
              "Tank"
            ]
          }
        },
        {
          "value": {
            "id": 119,
            "name": "Draven",
            "blurb": "Unlike his brother Darius, victory in battle was never enough for Draven. He craved recognition, acclaim, and glory. He first sought greatness in the Noxian military, but his flair for the dramatic went severely underappreciated. Thirsting for a method...",
            "tags": [
              "Marksman"
            ]
          }
        },
        {
          "value": {
            "id": 58,
            "name": "Renekton",
            "blurb": "''Blood and vengeance.''<br><br>Renekton is a terrifying, rage-fueled Ascended being from the scorched deserts of Shurima. Once, he was his empire's most esteemed warrior, leading the armies of Shurima to countless victories. However, after the empire's...",
            "tags": [
              "Fighter",
              "Tank"
            ]
          }
        },
        {
          "value": {
            "id": 82,
            "name": "Mordekaiser",
            "blurb": "''All things must die... and yet I live on.''<br><br>The baleful revenant Mordekaiser is among the most terrifying and hateful spirits haunting the Shadow Isles. He has existed for countless centuries, shielded from true death by necromantic sorcery and...",
            "tags": [
              "Fighter"
            ]
          }
        },
        {
          "value": {
            "id": 432,
            "name": "Bard",
            "blurb": "Bard travels through realms beyond the imagination of mortal beings. Some of Valoran's greatest scholars have spent their lives trying to understand the mysteries he embodies. This enigmatic spirit has been given many names throughout the history of...",
            "tags": [
              "Support",
              "Mage"
            ]
          }
        },
        {
          "value": {
            "id": 36,
            "name": "Dr. Mundo",
            "blurb": "''Beware the Madman of Zaun. In his eyes, you are already dead''<br><br>It is said that the man now known as Dr. Mundo was born without any sort of conscience. Instead, he had an unquenchable desire to inflict pain through experimentation. By the time...",
            "tags": [
              "Fighter",
              "Tank"
            ]
          }
        },
        {
          "value": {
            "id": 30,
            "name": "Karthus",
            "blurb": "''Death is not the end of the journey, it is just the beginning...''<br><br>The harbinger of oblivion, Karthus is an undying spirit whose haunting songs are a prelude to the horror of his nightmarish appearance. The living fear the eternity of undeath...",
            "tags": [
              "Mage"
            ]
          }
        },
        {
          "value": {
            "id": 51,
            "name": "Caitlyn",
            "blurb": "''Go ahead, run. I'll give you a five minute head start.''<br><br>One of the reasons Piltover is known as the City of Progress is because it has an extraordinarily low crime rate. This hasn't always been the case; brigands and thieves of all sorts used...",
            "tags": [
              "Marksman"
            ]
          }
        },
        {
          "value": {
            "id": 19,
            "name": "Warwick",
            "blurb": "Warwick is a monster who hunts the gray alleys of Zaun. Transformed by agonizing experiments, his body is fused with an intricate system of chambers and pumps, machinery filling his veins with alchemical rage. Bursting out of the shadows, he preys upon...",
            "tags": [
              "Fighter",
              "Tank"
            ]
          }
        },
        {
          "value": {
            "id": 107,
            "name": "Rengar",
            "blurb": "On every wall of his den, the trophy hunter Rengar mounts the heads, horns, claws, and fangs of the most lethal creatures in Valoran. Though his collection is extensive, he remains unsatisfied, tirelessly seeking greater game. He takes time with every...",
            "tags": [
              "Assassin",
              "Fighter"
            ]
          }
        },
        {
          "value": {
            "id": 4,
            "name": "Twisted Fate",
            "blurb": "Twisted Fate is an infamous card sharp and swindler who has gambled and charmed his way across much of the known world, earning the enmity and admiration of the rich and foolish alike. He rarely takes things seriously, greeting each day with a mocking...",
            "tags": [
              "Mage"
            ]
          }
        },
        {
          "value": {
            "id": 126,
            "name": "Jayce",
            "blurb": "Armed with wit, charm, and his signature transforming hammer, Jayce lives to protect his native Piltover. Long before his nation called him a hero, however, he was a promising young inventor. When Piltover commissioned him to study a rare arcane crystal...",
            "tags": [
              "Fighter",
              "Marksman"
            ]
          }
        },
        {
          "value": {
            "id": 136,
            "name": "Aurelion Sol",
            "blurb": "Aurelion Sol once graced the vast emptiness of the cosmos with celestial wonders of his own devising. Now, he is forced to wield his awesome power at the behest of a space-faring empire that tricked him into servitude. Desiring a return to his...",
            "tags": [
              "Mage",
              "Fighter"
            ]
          }
        },
        {
          "value": {
            "id": 268,
            "name": "Azir",
            "blurb": "''Shurima was once the glory of Runeterra. I will make it so again.''<br><br>Azir was a mortal emperor of Shurima in a far distant age, a proud man who stood at the cusp of immortality. His hubris saw him betrayed and murdered at the moment of his...",
            "tags": [
              "Mage",
              "Marksman"
            ]
          }
        },
        {
          "value": {
            "id": 79,
            "name": "Gragas",
            "blurb": "The only thing more important to Gragas than fighting is drinking. His unquenchable thirst for stronger ale has led him in search of the most potent and unconventional ingredients to toss in his still. Impulsive and unpredictable, this rowdy carouser...",
            "tags": [
              "Fighter",
              "Mage"
            ]
          }
        },
        {
          "value": {
            "id": 427,
            "name": "Ivern",
            "blurb": "Ivern Bramblefoot, known to many as the Green Father, is a peculiar half man, half tree who roams Runeterra's forests, cultivating life everywhere he goes. He knows the secrets of the natural world, and holds deep friendships with all things that grow...",
            "tags": [
              "Support",
              "Mage"
            ]
          }
        },
        {
          "value": {
            "id": 131,
            "name": "Diana",
            "blurb": "''I am the light coursing in the soul of the moon.''<br><br>Bearing her crescent moonblade, Diana fights as a warrior of the Lunari, a faith all but quashed in the lands around Mount Targon. Clad in shimmering armor the color of winter snow at night...",
            "tags": [
              "Fighter",
              "Mage"
            ]
          }
        },
        {
          "value": {
            "id": 15,
            "name": "Sivir",
            "blurb": "''I don't care what face is on your coin, as long as it pays.''<br><br>Sivir is a renowned fortune hunter and mercenary captain who plies her trade in the deserts of Shurima. Armed with her legendary jeweled crossblade, she has fought and won countless...",
            "tags": [
              "Marksman"
            ]
          }
        },
        {
          "value": {
            "id": 163,
            "name": "Taliyah",
            "blurb": "Taliyah is a nomadic mage from Shurima who weaves stone with energetic enthusiasm and raw determination. Torn between teenage wonder and adult responsibility, she has crossed nearly all of Valoran on a journey to learn the true nature of her growing...",
            "tags": [
              "Mage",
              "Support"
            ]
          }
        },
        {
          "value": {
            "id": 68,
            "name": "Rumble",
            "blurb": "''Ugh, it's gonna take forever to scrape your face off my suit!''<br><br>Even amongst yordles, Rumble was always the runt of the litter. As such, he was used to being bullied. In order to survive, he had to be scrappier and more resourceful than his...",
            "tags": [
              "Fighter",
              "Mage"
            ]
          }
        },
        {
          "value": {
            "id": 62,
            "name": "Wukong",
            "blurb": "During the chaos of the Rune Wars, an enormous runestone was lost deep within the Plague Jungles. It remained there, untouched for centuries, emanating a potent magic which infused nearby wildlife with sentience and vitality. A group of monkeys who were...",
            "tags": [
              "Fighter",
              "Tank"
            ]
          }
        },
        {
          "value": {
            "id": 90,
            "name": "Malzahar",
            "blurb": "Many men have gone mad beneath the glare of the Shurima sun, but it was during the night's chilling embrace that Malzahar relinquished his sanity. Malzahar was born a seer, blessed with the gift of prophecy. His talent, though unrefined, promised to be...",
            "tags": [
              "Mage",
              "Assassin"
            ]
          }
        },
        {
          "value": {
            "id": 111,
            "name": "Nautilus",
            "blurb": "Once, Nautilus was a sailor commissioned by the Institute of War to explore the uncharted reaches of the Guardian's Sea. This expedition took him deep into unknown waters where he and his crew found a vast section of black oozing liquid that none of the...",
            "tags": [
              "Tank",
              "Fighter"
            ]
          }
        },
        {
          "value": {
            "id": 254,
            "name": "Vi",
            "blurb": "To Vi, every problem is just another brick wall to punch through with her gigantic hextech gauntlets. Though she grew up on the wrong side of the law, Vi now uses her criminal know-how to serve Piltover's police force. Vi's brash attitude, abrasive...",
            "tags": [
              "Fighter",
              "Assassin"
            ]
          }
        },
        {
          "value": {
            "id": 122,
            "name": "Darius",
            "blurb": "There is no greater symbol of Noxian might than Darius, the nation's most feared and battle-hardened warrior. Orphaned at a young age, Darius had to fight to keep himself and his younger brother alive. By the time he joined the military, he had already...",
            "tags": [
              "Fighter",
              "Tank"
            ]
          }
        },
        {
          "value": {
            "id": 83,
            "name": "Yorick",
            "blurb": "''These isles… How they scream.''<br>The last survivor of a long-forgotten religious order, Yorick is both blessed and cursed with power over the dead. Trapped on the Shadow Isles, his only companions are the rotting corpses and shrieking spirits that...",
            "tags": [
              "Fighter",
              "Tank"
            ]
          }
        },
        {
          "value": {
            "id": 222,
            "name": "Jinx",
            "blurb": "Jinx lives to wreak havoc without a thought for the consequences, leaving a trail of mayhem and panic in her wake. A manic and impulsive criminal, she despises nothing more than boredom, and gleefully brings her own volatile brand of pandemonium to the...",
            "tags": [
              "Marksman"
            ]
          }
        }
      ]

champions = []

for dic in raw:
    champions.append(dic['value']['name'])
    
for champion in champions:
    fetch_img_urls(champion, browser, 1, 30)

