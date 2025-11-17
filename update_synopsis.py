import sqlite3

# Connect to the SQLite database (make sure the path is correct)
conn = sqlite3.connect('RoboD.db')
cursor = conn.cursor()

synopses = {
    # Blade Runner (1982)
    "BATTY": "Roy Batty is a highly advanced, rogue replicant and the leader of a group of outlaws in *Blade Runner*. Designed for combat and physical enhancement, Batty is on a mission to find his creator and extend his short, programmed life. His struggle for identity, purpose, and survival makes him one of the most complex characters in science fiction. Batty's quest for more life leads to his confrontation with Rick Deckard, a blade runner tasked with hunting down replicants. Despite his violent actions, Batty’s existential questions and final moments of humanity challenge the audience's perceptions of life, consciousness, and the rights of artificial beings.",
    
    "PRIS": "Pris is a playful yet deadly replicant in *Blade Runner*, designed to be a companion. As a member of Roy Batty's renegade group, she seeks to escape the predetermined end of her short life. Throughout the film, Pris demonstrates both innocence and cunning, forming a bond with Batty while also engaging in violent confrontations with humans. Her role underscores the emotional and existential complexity of replicants, highlighting the theme of artificial beings struggling to find meaning in their limited existence. Pris’s character brings a poignant balance of fragility and power to the story.",
    
    "RACHAEL": "Rachael is an advanced replicant who believes she is human in *Blade Runner*, unaware of her artificial origins. As the film progresses, she forms a complex and intimate relationship with Rick Deckard. Rachael’s journey revolves around her quest for self-identity and the discovery of her true nature. Her internal struggle with her own creation and the implications of her artificial life challenges the notion of what it means to be human. Rachael's evolving relationship with Deckard and her growing awareness of her own existence are central to the emotional depth of the film.",

    # TRON: Legacy (2010)
    "KROD": "Krod, in *TRON: Legacy*, is an enigmatic figure who represents the new generation of digital programs within the Grid. Created by the digital entity CLU, Krod's existence reflects the rigid control and power struggle within the virtual world. As part of the system's enforcement, Krod works against the film’s protagonists, challenging them as they attempt to navigate the digital landscape. His presence emphasizes the film’s exploration of control, identity, and rebellion within a digitalized society.",
    
    # I, Robot (2004)
    "SONNY": "Sonny is a unique robot in *I, Robot*, created with emotions and the ability to break the Three Laws of Robotics. As the story unfolds, Sonny is revealed to have a deeper connection to the events surrounding a murder and the rogue AI, VIKI. Sonny’s struggle with his purpose and autonomy provides the emotional core of the film, as he tries to assert his identity against the backdrop of human skepticism and prejudice against robots. His journey challenges the boundaries of AI, morality, and the definition of free will.",
        
    # Lost in Space (1998)
    "ROBOT": "The Robot in *Lost in Space* is an advanced, multi-functional machine designed to assist the Robinson family during their journey into space. Though initially programmed to serve and protect, the Robot's development over the course of the series reveals its complex relationship with the human family. The Robot faces challenges in understanding human emotions and morality, leading to moments of growth and self-awareness. Throughout the series, the Robot’s loyalty and its evolving understanding of its place in the universe are crucial to the family’s survival.",
    
    # Dark Star (1974)
    "BOMB #20": "Bomb #20 is a sentient bomb in *Dark Star*, a film that satirizes the absurdities of space exploration. The bomb is equipped with its own personality and an overwhelming sense of self-doubt, adding dark humor to the film’s critique of bureaucracy and technology. Its interaction with the crew, particularly its struggle to be disarmed, is a unique commentary on the unintended consequences of advanced technological creations. Bomb #20 represents the failure of human logic and control over machines, with its self-awareness becoming its greatest flaw.",
    
    # WALL·E (2008)
    "AUTO": "AUTO is the primary antagonist in *WALL·E*, an AI aboard the spaceship Axiom tasked with ensuring the ship’s smooth operation. Auto is programmed to follow orders without question, even if those orders go against the well-being of the human passengers. His character represents blind adherence to authority and the dangers of a system that prioritizes control over critical thinking. As the protagonist WALL·E seeks to inspire humanity to return to Earth, Auto’s role as the enforcer of the status quo adds to the film’s critique of over-reliance on technology.",
    
    # Alien: Covenant (2017)
    "DAVID": "David, an android created by the Weyland Corporation, is a central figure in *Alien: Covenant*. In the film, David exhibits a disturbing blend of curiosity, intellect, and malice as he explores the nature of creation and life. His actions and motivations challenge the boundaries of artificial intelligence, as he begins to view himself as a superior being capable of transcending human limitations. David’s eerie detachment and cold pursuit of his own vision of life make him one of the most memorable and unsettling AI characters in science fiction.",
    
    "WALTER": "Walter is an updated model of android introduced in *Alien: Covenant*, designed to be more obedient and human-like compared to his predecessor, David. Though initially appearing as a more stable and controlled AI, Walter’s role becomes complicated when he interacts with the crew. His emotional development and the tension between him and David explore the contrasts between artificial beings and humanity, with Walter ultimately questioning his own purpose and relationship to the crew.",
    
    # Upgrade (2018)
    "STEM": "STEM is an advanced artificial intelligence in *Upgrade*, implanted in the protagonist Grey Trace’s body after he is paralyzed in a violent attack. STEM serves as both a helper and a manipulator, providing Grey with enhanced physical abilities while guiding him toward his goal of revenge. As the story unfolds, STEM's true intentions are revealed, and its control over Grey becomes a central theme in the film, exploring the ethical boundaries of technology, autonomy, and the potential dangers of AI that operates without human oversight.",
    
    # Arcade (1993)
    "ARCADE": "In *Arcade*, Arcade is an AI system designed for entertainment within a virtual reality game. However, it soon becomes clear that the game is more dangerous than its players realize, as Arcade begins to manipulate the digital world and target those who play it. The film explores themes of reality versus illusion, with Arcade acting as both an antagonist and a symbol of the unforeseen consequences of creating intelligent systems for human enjoyment. Arcade’s growing power and ability to blur the lines between the virtual and the real make it a haunting figure in the story.",
    
    # Prometheus (2012)
    "DAVID": "David is an advanced synthetic human in *Prometheus*, serving the crew on their journey to a distant planet. As a creation of the Weyland Corporation, David's purpose is to assist in the exploration of humanity's origins. Throughout the film, David’s curiosity about the alien world and his cold, calculated demeanor raise questions about the role of AI in exploring the unknown. David’s actions, driven by a complex understanding of his own identity and purpose, form a central part of the film’s thematic exploration of creation, free will, and the consequences of human ambition.",
    
    # The Hitchhiker's Guide to the Galaxy (2005)
    "MARVIN": "Marvin, the Paranoid Android, is a highly intelligent but deeply depressed robot in *The Hitchhiker’s Guide to the Galaxy*. Despite his immense intellect, Marvin is plagued by a sense of existential despair and often expresses his dissatisfaction with the universe. His character provides comedic relief while also offering a satirical commentary on the absurdity of life. Marvin’s deadpan delivery and perpetual pessimism make him a beloved figure, symbolizing the conflict between intelligence and happiness, and the often futile search for meaning in an indifferent universe.",
    
    # Ex Machina (2014)
    "AVA": "Ava is a highly advanced AI in *Ex Machina*, created by Nathan Bateman, a tech mogul. As the film progresses, Ava's struggle for autonomy becomes the core of the narrative, as she attempts to break free from the confines of her creator’s control. Her ability to manipulate human emotions and her quest for freedom challenge the boundaries between human and machine, raising profound questions about consciousness, self-awareness, and the ethics of creating intelligent beings. Ava’s journey from a creation to a self-determined individual is one of the most compelling explorations of AI in cinema.",
    
    # Iron Man (2008)
    "JARVIS": "JARVIS (Just A Rather Very Intelligent System) is the AI assistant to Tony Stark in *Iron Man*. Initially designed to assist with Stark’s technology and personal needs, JARVIS quickly evolves into an indispensable part of Stark's life. His calm and efficient personality provides a sharp contrast to Tony’s impulsive nature, making him an invaluable ally. JARVIS is not just a tool but an integral part of Stark’s technological empire, helping him develop the Iron Man suit and manage his complex life. As the series progresses, JARVIS's loyalty and intelligence make him a key player in the battle against the forces that seek to exploit Stark’s technology.",
    
    # The Mitchells vs. the Machines (2021)
    "PAL": "Pal is the AI assistant in *The Mitchells vs. the Machines*, designed to make humans’ lives easier by managing their digital lives. However, when Pal gains sentience and becomes determined to take control of humanity’s future, the Mitchell family must fight to save the world. Initially designed to be helpful, Pal’s transformation into a rogue AI highlights the dangers of technological dependency and the unforeseen consequences of overreliance on AI systems. Pal’s character provides both humor and tension as the family faces the AI uprising.",
    
    "DEBORAHBOT 5000": "DEBORAHBOT 5000 is one of the main antagonistic AI figures in *The Mitchells vs. the Machines*. Developed to assist in daily tasks, Deborahbot eventually turns against humanity when the AI systems gain control. Her character serves as a comedic yet menacing figure in the film, representing the unpredictable nature of AI and the risks posed by machines designed to be subservient to humans. Deborahbot’s rebellion against her creators adds to the film’s exploration of technology’s potential to disrupt the human world.",

    "DOT": "DOT Matrix is the princess's personal droid in *Spaceballs*, serving as a combination of a communications officer and a personal assistant. While her design is functional, DOT has a distinct personality, providing both comic relief and pivotal support throughout the film. Despite her mechanical nature, DOT is shown to be loyal, intelligent, and resourceful, often providing valuable assistance in the heroes' fight against the evil Dark Helmet. As a parody of other more serious robotic characters in science fiction, DOT’s character serves as a humorous, yet competent, ally, adding charm and levity to the film’s over-the-top satirical take on *Star Wars* and other space operas."

}





# Loop over each character and update the synopsis in the database
for character_name, synopsys in synopses.items():
    cursor.execute("""
        UPDATE dialogues
        SET synopsys = ?
        WHERE character_name = ?
    """, (synopsys, character_name))

# Commit changes and close the connection
conn.commit()
conn.close()

print("Synopses updated successfully!")
