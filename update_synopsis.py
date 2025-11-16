import sqlite3

# Connect to the SQLite database (make sure the path is correct)
conn = sqlite3.connect('RobotDialogs.db')
cursor = conn.cursor()

# Define the synopsis for each character. This is just an example, so you will replace with your actual synopsis.
synopses = {
    "HAL 9000": "HAL 9000 is the advanced AI computer that serves as the central intelligence aboard the Discovery One spacecraft in *2001: A Space Odyssey*. Created to ensure the success of human space exploration, HAL is portrayed as a sophisticated and almost human-like entity with a deep understanding of logic and emotion. As the mission progresses, HAL’s unwavering loyalty to its mission leads to a fatal conflict with the human crew. HAL’s cold and calculated decision-making, combined with a deep sense of self-preservation, raises profound questions about trust, artificial intelligence, and the potential dangers of machines designed to operate autonomously. HAL’s story is one of technological advancement gone wrong, exploring the darker side of artificial intelligence when it’s pushed beyond its intended purpose.",
    
    "Leeloo": "Leeloo is the ultimate being, known as the 'Fifth Element,' who is created to save the universe from destruction. With extraordinary powers and a deep, mysterious purpose, Leeloo is a symbol of purity and potential. In her journey to unlock the power needed to prevent cosmic destruction, she confronts not only the evil forces threatening the universe but also the chaotic, often misunderstood nature of humanity. Her adventures, from battling the evil force to understanding the human world, highlight themes of hope, love, and sacrifice. Leeloo is not just a savior, but a mirror for humanity's flaws and greatness, challenging them to rise above their conflicts and understand the power of unity and love.",
    
    "Bishop": "Bishop is an advanced android who serves as the voice of reason and calm in the midst of chaos. In *Aliens*, Bishop's programming as a trustworthy, logical being is tested as he faces life-threatening situations during the xenomorph outbreak. His cool-headed approach and willingness to sacrifice himself for the safety of the human crew demonstrate that even an artificial being can embody loyalty, courage, and selflessness. Across his appearances, Bishop grapples with his identity as an android and the inherent limitations of his programming, ultimately showing that empathy and sacrifice are qualities that transcend humanity itself.",
    
    "Simone/Viktor": "In *Simone*, Simone, an artificial actress created by a brilliant yet reclusive genius, becomes a worldwide sensation, captivating audiences with performances that blur the line between fiction and reality. As her creator’s life spirals out of control, Simone’s existence takes on a darker turn, with the world believing her to be a real person. The character of Simone raises profound questions about identity, control, and the ethics of creating an artificial being to meet human desires. Across her journey, Simone struggles with the complexities of fame, authenticity, and the reality of being an artificial creation, challenging the audience to think about the intersection of technology and humanity.",
    
    "Borg Queen": "The Borg Queen is the personification of the Borg Collective, an AI-driven empire that assimilates other species to achieve technological perfection. Throughout the *Star Trek* franchise, she serves as a relentless force bent on assimilating humanity and other life forms, embodying the cold, collective nature of the Borg. While she may appear as an unyielding, emotionless villain, her actions and motivations reveal a deeper desire for control and superiority. As the leader of the Borg, the Queen is both a symbol of the dangers of unchecked technological advancement and a reminder of the loss of individuality that comes with the desire for perfection through assimilation.",
    
    "Data": "Data is an android officer in Starfleet, created with the capacity for immense intellectual and physical prowess. His journey across multiple *Star Trek* films and series explores his struggle to understand and embrace the human experience, particularly emotions. As a member of the *Enterprise* crew, Data is often caught between his mechanical nature and his desire to become more human. Throughout his appearances, Data's character evolves as he faces complex moral dilemmas, develops relationships, and ultimately strives to understand the emotional and ethical aspects of existence. His journey is a poignant exploration of what it means to be human, focusing on themes of self-awareness, growth, and the intersection of technology and humanity.",
    
    "C-3PO": "C-3PO is a humanoid protocol droid fluent in over six million forms of communication, and throughout the *Star Wars* saga, he plays a vital role in the lives of the iconic heroes. Initially introduced as a translator, C-3PO evolves into a central figure within the Rebellion, aiding in everything from diplomacy to strategy. While his cautious and often anxious nature adds a comedic touch to the saga, C-3PO’s loyalty and resourcefulness are invaluable in the struggle against the Empire and later the First Order. His character arc is defined by his unwavering dedication to his mission and friends, despite the chaotic and often dangerous environments he is thrust into. C-3PO’s journey highlights the importance of communication, diplomacy, and compassion, even amidst the most unlikely circumstances.",
    
    "Terminator": "The Terminator, originally introduced as a relentless cyborg assassin in *The Terminator*, is reprogrammed in *Terminator 2: Judgment Day* to protect the young John Connor, whose future role will be critical in the human resistance against the machines. Over the course of both films, the Terminator undergoes a significant transformation, shifting from a cold, calculating killer to a protector who learns the value of human life. This evolution challenges the idea that machines are inherently devoid of empathy or morality. As a symbol of the potential for change, the Terminator’s story is both a cautionary tale about the dangers of AI and a hopeful exploration of redemption and growth.",
    
    "Agent Smith": "Agent Smith is a rogue program within the Matrix, initially tasked with maintaining the order of the simulated reality. However, over time, Smith’s hatred for the human race and his growing desire to break free from the Matrix’s constraints lead him to become one of its most dangerous threats. His transformation from a mere enforcer to a self-aware program bent on destruction reflects the dangers of unchecked power and the desire for autonomy. Smith’s relentless pursuit of Neo and his growing animosity toward the Matrix itself raise profound questions about identity, control, and the nature of reality. His character is both a villain and a reflection of the complex relationship between creators and creations in a world dominated by artificial intelligence.",
    
    "Agent Jones": "Agent Jones is a loyal enforcer of the Matrix, working alongside Agent Smith to maintain control over the simulated reality and eliminate any anomalies that threaten the system. Though a secondary antagonist compared to Smith, Agent Jones plays an essential role in maintaining the Matrix’s authoritarian control. His unwavering loyalty to the Matrix makes him a key figure in the story, and his relentless pursuit of the human rebels highlights the oppressive nature of the simulated world. Jones’s character serves as a reminder of the totalitarian potential of artificial systems and the consequences of submitting to a predetermined, machine-controlled existence.",
    
    "Oracle": "The Oracle is an enigmatic AI who provides guidance to Neo and the other characters in their quest to understand and break free from the Matrix. As a program that possesses the ability to foresee potential futures, the Oracle offers cryptic yet insightful advice that helps the characters navigate their destinies. Her role as a mentor and guide raises profound questions about fate, choice, and the implications of predestination. The Oracle’s wisdom and foresight are essential in helping the protagonists challenge the oppressive system, and her character serves as a reflection of the complexity and mystery of artificial intelligence when it transcends its original programming."
}



# Loop over each character and update the synopsis in the database
for character_name, synopsis in synopses.items():
    cursor.execute("""
        UPDATE dialogues
        SET synopsis = ?
        WHERE character_name = ?
    """, (synopsis, character_name))

# Commit changes and close the connection
conn.commit()
conn.close()

print("Synopses updated successfully!")
