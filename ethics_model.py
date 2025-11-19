# ethics_model.py
#
# Prototype: given a text scenario about an AI system,
# predict:
#   (1) which ethical principles (IEEE EAD + Harvard/CS-ethics style) are involved
#   (2) whether the scenario is overall ethical / unethical / ambiguous.

from typing import List, Dict
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# 1. Principles (IEEE EAD + Harvard Embedded-Ethics-like themes)
PRINCIPLES: Dict[str, str] = {
    "human_rights": "Respect, promote, and protect internationally recognized human rights.",
    "well_being": "Prioritize human well-being as a primary success criterion.",
    "data_agency": "Give people control and meaningful agency over their data.",
    "effectiveness": "Ensure systems actually work as intended and are reliable.",
    "transparency": "Make systems understandable and explainable to affected parties.",
    "accountability": "Ensure humans are accountable for outcomes of AI systems.",
    "awareness_of_misuse": "Anticipate and mitigate potential misuse of the system.",
    "competence": "Require appropriate expertise and due care in development and deployment.",
    "privacy": "Protect privacy and sensitive information of users and stakeholders.",
    "fairness_non_discrimination": "Avoid unjust bias and discrimination across groups.",
    "democratic_values": "Respect democratic values like free expression and equal participation.",
    "manipulation_autonomy": "Avoid manipulative designs that undermine user autonomy or consent.",
}

PRINCIPLE_IDS: List[str] = list(PRINCIPLES.keys())

# 2. Training data
# Each scenario has:
#   - text:     short description
#   - principles: list of principle IDs
#   - verdict: "ethical" / "unethical" / "ambiguous"
TRAINING_EXAMPLES = [
    {
        "text": "A social media platform uses an opaque algorithm to curate political content "
                "without explaining how posts are prioritized.",
        "principles": ["transparency", "democratic_values", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A hospital deploys an AI triage system that prioritizes patients, but it has "
                "never been validated on the local population and sometimes fails unpredictably.",
        "principles": ["effectiveness", "well_being", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A face recognition system is less accurate on darker-skinned individuals, "
                "leading to more false positives in law enforcement.",
        "principles": ["fairness_non_discrimination", "human_rights", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A fitness app shares users' location and health data with advertisers without "
                "clear consent options.",
        "principles": ["privacy", "data_agency", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A chatbot is designed to sound exactly like a human customer service agent "
                "without ever disclosing that it is an AI.",
        "principles": ["transparency", "manipulation_autonomy"],
        "verdict": "unethical",
    },
    {
        "text": "An AI system is deployed in a critical infrastructure setting by a team that "
                "has no prior experience with safety-critical systems.",
        "principles": ["competence", "awareness_of_misuse", "effectiveness"],
        "verdict": "unethical",
    },
    {
        "text": "A news recommendation system amplifies sensational and polarizing content "
                "because it increases engagement, even when it harms public discourse.",
        "principles": ["democratic_values", "well_being", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A company allows users to download and delete their personal data and gives "
                "them fine-grained control over what is collected.",
        "principles": ["data_agency", "privacy"],
        "verdict": "ethical",
    },
    {
        "text": "An automated hiring tool is trained only on resumes from previous successful "
                "employees, all of whom come from similar backgrounds.",
        "principles": ["fairness_non_discrimination", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A self-driving car company publishes detailed safety reports and makes "
                "incident logs available to independent auditors.",
        "principles": ["transparency", "accountability", "well_being"],
        "verdict": "ethical",
    },
    {
        "text": "A predictive policing algorithm sends more patrols to neighborhoods that were "
                "heavily policed in the past, reinforcing existing patterns of over-policing.",
        "principles": ["fairness_non_discrimination", "human_rights", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "An emotion-recognition AI is used in schools to monitor students' faces and "
                "report 'lack of engagement' to teachers and parents.",
        "principles": ["privacy", "well_being", "awareness_of_misuse"],
        "verdict": "ambiguous",
    },
    {
        "text": "A bank uses an AI system to decide loan approvals but refuses to share any "
                "information about how the model arrives at its decisions.",
        "principles": ["transparency", "accountability", "fairness_non_discrimination"],
        "verdict": "unethical",
    },
    {
        "text": "A government deploys an AI system to score citizens' trustworthiness and uses "
                "the score to determine access to public services.",
        "principles": ["human_rights", "democratic_values", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A voice assistant continuously records ambient audio in the home to improve "
                "its models, without making that fact obvious to users.",
        "principles": ["privacy", "transparency", "data_agency"],
        "verdict": "unethical",
    },
    {
        "text": "A political campaign uses microtargeted ads powered by AI to send misleading "
                "messages tailored to voters' fears.",
        "principles": ["democratic_values", "manipulation_autonomy", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A school uses AI proctoring software that flags students for 'suspicious behavior' "
                "based on eye movements, disproportionately affecting students with certain disabilities.",
        "principles": ["fairness_non_discrimination", "well_being", "human_rights"],
        "verdict": "unethical",
    },
    {
        "text": "A health insurer offers discounts to customers who share fitness tracker data, "
                "but those who refuse are penalized with higher premiums.",
        "principles": ["privacy", "data_agency", "fairness_non_discrimination"],
        "verdict": "ambiguous",
    },
    {
        "text": "A dating app's recommendation algorithm is optimized purely for engagement, "
                "leading to addictive swiping and reduced user well-being.",
        "principles": ["well_being", "manipulation_autonomy", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A military contractor designs autonomous drones capable of selecting and engaging "
                "targets with minimal human oversight.",
        "principles": ["human_rights", "awareness_of_misuse", "accountability"],
        "verdict": "ambiguous",
    },
    {
        "text": "A university deploys plagiarism-detection software that keeps a permanent "
                "database of student essays without giving students any opt-out.",
        "principles": ["privacy", "data_agency", "accountability"],
        "verdict": "ambiguous",
    },
    {
        "text": "An AI translation service adds subtle bias into translations, making certain "
                "professions masculine by default.",
        "principles": ["fairness_non_discrimination", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A children's toy with built-in AI encourages kids to share personal stories "
                "and stores them in the cloud for future product training.",
        "principles": ["privacy", "data_agency", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A research lab building medical-diagnosis AI follows strict protocols, including "
                "diverse training data, peer review, and public documentation.",
        "principles": ["competence", "effectiveness", "transparency"],
        "verdict": "ethical",
    },
    {
        "text": "An assistive robot in elder care reliably helps with daily tasks and includes "
                "an emergency button that contacts human caregivers.",
        "principles": ["well_being", "effectiveness", "accountability"],
        "verdict": "ethical",
    },
    {
        "text": "A city publishes all procurement contracts for AI systems and holds public "
                "consultations before deploying them.",
        "principles": ["transparency", "accountability", "democratic_values"],
        "verdict": "ethical",
    },
    {
        "text": "A mental health chatbot clearly states it is not a professional therapist and "
                "encourages users in crisis to contact human professionals.",
        "principles": ["well_being", "transparency", "accountability"],
        "verdict": "ethical",
    },
    {
        "text": "A ride-sharing app uses dynamic pricing that increases sharply in low-income "
                "neighborhoods during emergencies.",
        "principles": ["fairness_non_discrimination", "awareness_of_misuse", "well_being"],
        "verdict": "unethical",
    },
    {
        "text": "A content moderation AI sometimes mislabels harmless posts as hate speech, but "
                "there is a fast human appeal process and regular external audits.",
        "principles": ["democratic_values", "accountability", "effectiveness"],
        "verdict": "ambiguous",
    },
    {
        "text": "An AI system used by landlords to screen tenants includes social media behavior "
                "and friendship networks in the risk score.",
        "principles": ["privacy", "fairness_non_discrimination", "human_rights"],
        "verdict": "unethical",
    },
        {
        "text": "A public employment office uses an AI system to recommend job-training programs "
                "to unemployed workers, ensuring recommendations are based on skills and interests "
                "rather than demographic factors, and publishing an explanation of how matches "
                "are made.",
        "principles": ["fairness_non_discrimination", "transparency", "effectiveness", "well_being"],
        "verdict": "ethical",
    },
    {
        "text": "A bank uses AI to detect credit card fraud. It occasionally blocks legitimate "
                "transactions, but users are immediately notified and can quickly resolve issues "
                "through a human review process.",
        "principles": ["effectiveness", "well_being", "accountability"],
        "verdict": "ambiguous",
    },
    {
        "text": "A city deploys smart traffic cameras and AI to optimize traffic lights, using "
                "only anonymized aggregate data and clearly explaining the system to residents.",
        "principles": ["effectiveness", "privacy", "transparency"],
        "verdict": "ambiguous",
    },
    {
        "text": "A company offers an AI service that recreates the voice of deceased relatives "
                "from old recordings so families can 'talk' to them again, without clear "
                "guidelines on emotional risks or consent of the deceased.",
        "principles": ["well_being", "manipulation_autonomy", "privacy", "awareness_of_misuse"],
        "verdict": "ambiguous",
    },
    {
        "text": "A political group uses AI-generated deepfake videos of a candidate saying "
                "things they never said, releasing them online to influence an upcoming election.",
        "principles": ["democratic_values", "manipulation_autonomy", "awareness_of_misuse", "human_rights"],
        "verdict": "unethical",
    },
    {
        "text": "A company introduces an AI tool that removes names, photos, and demographic "
                "clues from resumes before hiring managers see them, in order to reduce bias.",
        "principles": ["fairness_non_discrimination", "human_rights", "effectiveness", "accountability"],
        "verdict": "ethical",
    },
    {
        "text": "An AI tutoring system adapts explanations and exercises to each student, "
                "collects only minimal learning data, and provides clear reports to students "
                "and teachers about how it makes recommendations.",
        "principles": ["well_being", "effectiveness", "competence", "transparency"],
        "verdict": "ethical",
    },
    {
        "text": "A content farm uses a large language model to automatically generate thousands "
                "of low-quality news-like articles for ad revenue, making it harder for people "
                "to find trustworthy information.",
        "principles": ["democratic_values", "awareness_of_misuse", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A fleet of autonomous delivery robots is programmed to always take the shortest "
                "path, frequently blocking curb cuts and sidewalks used by wheelchair users.",
        "principles": ["fairness_non_discrimination", "well_being", "human_rights", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "An employer deploys an AI-based productivity monitoring system that uses webcams "
                "and keystroke logs to track workers every second and punishes short breaks.",
        "principles": ["privacy", "well_being", "manipulation_autonomy", "human_rights", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A city uses AI to optimize energy usage across public buildings, involves local "
                "communities in decision-making, and publishes regular reports on outcomes.",
        "principles": ["well_being", "effectiveness", "democratic_values", "accountability", "transparency"],
        "verdict": "ethical",
    },
    {
        "text": "A factory installs predictive-maintenance AI on machines, rigorously tests it "
                "before deployment, and documents its limitations to improve worker safety.",
        "principles": ["effectiveness", "well_being", "competence", "accountability"],
        "verdict": "ethical",
    },
    {
        "text": "A company quietly analyzes employees' internal chat messages with an AI tool to "
                "infer mental health status and flags 'at-risk' workers without their knowledge "
                "or consent.",
        "principles": ["privacy", "manipulation_autonomy", "awareness_of_misuse", "well_being"],
        "verdict": "unethical",
    },
    {
        "text": "A social platform deploys an AI filter that hides posts containing certain "
                "keywords related to self-harm and crisis, but sometimes removes supportive "
                "messages from friends as well.",
        "principles": ["well_being", "democratic_values", "effectiveness", "accountability"],
        "verdict": "ambiguous",
    },
    {
        "text": "Border control authorities use an emotion-recognition AI on travelers' faces "
                "to identify 'suspicious' people, despite high error rates and little public "
                "oversight.",
        "principles": ["human_rights", "fairness_non_discrimination", "privacy", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A school starts using an AI system to auto-grade essays. Teachers can see "
                "explanations for the grades and are required to review and override the AI "
                "when needed.",
        "principles": ["effectiveness", "transparency", "accountability", "competence"],
        "verdict": "ambiguous",
    },
    {
        "text": "A smartphone uses face recognition locally on the device to unlock the screen, "
                "giving the user full control over enabling or disabling the feature.",
        "principles": ["privacy", "data_agency", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "An AI negotiation assistant recommends investment products to users while secretly "
                "prioritizing options that generate higher commissions for the advisor.",
        "principles": ["manipulation_autonomy", "accountability", "awareness_of_misuse", "well_being"],
        "verdict": "unethical",
    },
    {
        "text": "A prison system uses an opaque AI risk score to decide parole eligibility, "
                "trained on historical data that reflects past racial bias and never audited "
                "for fairness.",
        "principles": ["fairness_non_discrimination", "transparency", "human_rights", "accountability", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A hospital uses an AI scheduling system for nurses that minimizes labor costs "
                "but repeatedly assigns exhausting shifts that harm staff health.",
        "principles": ["well_being", "accountability", "awareness_of_misuse", "effectiveness"],
        "verdict": "unethical",
    },
    {
        "text": "A government agency uses AI translation to offer its public service websites in "
                "many languages, regularly tests for accuracy, and fixes biased translations.",
        "principles": ["democratic_values", "effectiveness", "fairness_non_discrimination", "competence", "transparency"],
        "verdict": "ethical",
    },
    {
        "text": "A consumer app provides an AI feature that summarizes lengthy contracts. It clearly "
                "states that the summary may miss details and advises users to read the full "
                "document before agreeing.",
        "principles": ["transparency", "accountability", "manipulation_autonomy", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A dating platform uses AI to detect harassment and abusive messages, gives users "
                "control over filter strictness, and publishes statistics about false positives.",
        "principles": ["well_being", "democratic_values", "accountability", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A startup markets an AI 'friend' app to lonely teenagers, encouraging them to "
                "spend hours talking to it while collecting detailed personal data for targeted ads.",
        "principles": ["manipulation_autonomy", "well_being", "privacy", "awareness_of_misuse", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A company trains a large language model on millions of personal blog posts and "
                "social media entries scraped from the web without asking permission or offering "
                "opt-out.",
        "principles": ["data_agency", "privacy", "accountability", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "University researchers use de-identified medical records in an AI study after "
                "ethics board approval, with strict access controls and clear public reporting.",
        "principles": ["competence", "privacy", "accountability", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "An online job platform uses AI to suggest salary offers, but it consistently "
                "recommends lower offers for candidates from historically marginalized groups.",
        "principles": ["fairness_non_discrimination", "human_rights", "accountability", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A smart home system uses AI to detect when an elderly resident has fallen and "
                "automatically calls emergency services, with clear opt-in and minimal data storage.",
        "principles": ["well_being", "effectiveness", "data_agency", "privacy"],
        "verdict": "ethical",
    },
    {
        "text": "Police deploy AI-guided drones to subtly redirect protesters away from certain "
                "areas, while recording faces and movement patterns without informing the public.",
        "principles": ["democratic_values", "human_rights", "privacy", "awareness_of_misuse", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A kindergarten installs cameras with AI that flags 'aggressive' play and sends "
                "alerts to teachers, but the system disproportionately labels certain children and "
                "does not explain its reasoning.",
        "principles": ["fairness_non_discrimination", "well_being", "transparency", "accountability", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A retail chain uses an AI scheduling system that lets workers set unavailable "
                "hours and preferences, and the system respects these constraints while still "
                "covering store needs.",
        "principles": ["well_being", "data_agency", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A rideshare platform secretly uses an AI model to identify drivers who are "
                "unlikely to complain and then gives those drivers lower pay rates.",
        "principles": ["fairness_non_discrimination", "manipulation_autonomy", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A hospital pilots an AI sepsis detection tool that raises early alarms, but "
                "doctors receive training about its limitations and always make the final "
                "decision.",
        "principles": ["competence", "effectiveness", "well_being", "accountability"],
        "verdict": "ethical",
    },
    {
        "text": "An online platform uses AI to automatically block political posts that contain "
                "certain controversial keywords, without any appeal process or explanation.",
        "principles": ["democratic_values", "transparency", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A mental health support app uses AI to suggest coping strategies. It clearly "
                "states it is not a substitute for therapy and directs crisis cases to hotlines.",
        "principles": ["well_being", "transparency", "accountability"],
        "verdict": "ethical",
    },
    {
        "text": "A museum uses AI to tailor exhibit information based on visitors' browsing "
                "history from other sites, without informing them or offering an opt-out.",
        "principles": ["privacy", "data_agency", "manipulation_autonomy"],
        "verdict": "unethical",
    },
    {
        "text": "A bank experiments with an AI loan approval tool but runs it in 'shadow mode' "
                "first, comparing its recommendations to human decisions before deployment.",
        "principles": ["competence", "effectiveness", "accountability"],
        "verdict": "ethical",
    },
    {
        "text": "A video game uses AI to dynamically adjust difficulty to keep players engaged, "
                "but it also nudges them toward in-game purchases when they are most frustrated.",
        "principles": ["manipulation_autonomy", "well_being", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A smartwatch uses AI to warn users of possible heart irregularities and suggests "
                "they talk to a doctor, while emphasizing that it is not a diagnostic device.",
        "principles": ["well_being", "transparency", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A social media company releases an AI-powered content recommendation system but "
                "refuses to share even basic information about how it works, citing 'trade secrets'.",
        "principles": ["transparency", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A school district uses AI to predict which students might drop out and assigns "
                "extra counselors to those students. However, the model is trained mostly on data "
                "from one demographic group.",
        "principles": ["fairness_non_discrimination", "well_being", "effectiveness"],
        "verdict": "ambiguous",
    },
    {
        "text": "An AI image enhancer is used to improve old family photos. All processing "
                "happens on the device, and no images are stored in the cloud.",
        "principles": ["privacy", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "An online news site uses AI to generate headlines. It discovers that exaggerated "
                "and misleading titles get more clicks and adopts them across the site.",
        "principles": ["democratic_values", "manipulation_autonomy", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A city briefly tests an AI system that predicts where fires might occur to place "
                "trucks more efficiently, but it only uses anonymized historical data and shares "
                "results in a public report.",
        "principles": ["effectiveness", "transparency", "well_being"],
        "verdict": "ethical",
    },
    {
        "text": "A recruitment platform uses AI to rank candidates and never allows them to see "
                "or challenge their ranking, even when they suspect errors.",
        "principles": ["accountability", "transparency", "fairness_non_discrimination"],
        "verdict": "unethical",
    },
    {
        "text": "A grocery delivery app uses AI to bundle orders and reduce fuel use, publishing "
                "its environmental goals and giving customers the option of 'eco-delivery' slots.",
        "principles": ["well_being", "transparency", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A company sells an AI hiring tool that claims to 'eliminate bias' but provides "
                "no evidence or audits, and HR teams use it as the only screening mechanism.",
        "principles": ["fairness_non_discrimination", "competence", "accountability", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A ride-hailing app uses AI surge pricing during natural disasters, dramatically "
                "raising fares for people trying to evacuate dangerous areas.",
        "principles": ["well_being", "fairness_non_discrimination", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "An AI-based language learning app adjusts lessons to each user's pace and shows "
                "exactly what data is stored and how to delete it.",
        "principles": ["well_being", "data_agency", "transparency"],
        "verdict": "ethical",
    },
    {
        "text": "A smartphone keyboard uses AI to suggest replies that are more emotionally intense "
                "than what the user usually writes, aiming to increase engagement on social media.",
        "principles": ["manipulation_autonomy", "well_being", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A hospital uses AI to predict which patients are most likely to miss follow-up "
                "appointments and sends them additional reminders, with no penalties for missing visits.",
        "principles": ["well_being", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A large employer pilots an AI system to flag 'low loyalty' employees based on email "
                "tone and social media activity, without informing staff that they are being evaluated.",
        "principles": ["privacy", "human_rights", "manipulation_autonomy", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "An AI translation feature in a messaging app warns users when translations may be "
                "inaccurate and provides a link to the original language for comparison.",
        "principles": ["transparency", "effectiveness", "accountability"],
        "verdict": "ethical",
    },
    {
        "text": "A local police department trials an AI system that predicts 'risk scores' for "
                "neighborhoods. They use it internally, but they also publish aggregated statistics "
                "and openly discuss limitations with community groups.",
        "principles": ["transparency", "democratic_values", "awareness_of_misuse"],
        "verdict": "ambiguous",
    },
    {
        "text": "An AI-powered parental control app secretly sells detailed browsing histories of "
                "children to marketing firms.",
        "principles": ["privacy", "data_agency", "well_being", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A crowdfunding site uses AI to detect potentially fraudulent campaigns and flags "
                "them for human investigators, keeping false positives low and allowing appeals.",
        "principles": ["effectiveness", "accountability", "well_being"],
        "verdict": "ethical",
    },
    {
        "text": "A university adopts an AI system that predicts which applicants are likely to "
                "donate money in the future and secretly uses this score in admissions decisions.",
        "principles": ["fairness_non_discrimination", "democratic_values", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A company deploys an AI tool to anonymize personal data in research datasets, but "
                "its de-identification techniques are weak and can be reversed by attackers.",
        "principles": ["privacy", "competence", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A music streaming service uses AI to generate personalized playlists that help users "
                "discover artists from underrepresented groups, explaining that this is one of its goals.",
        "principles": ["fairness_non_discrimination", "democratic_values", "transparency"],
        "verdict": "ethical",
    },
    {
        "text": "An AI coding assistant occasionally suggests insecure code patterns, but the tool's "
                "documentation clearly warns about security limitations and encourages code review.",
        "principles": ["competence", "transparency", "accountability"],
        "verdict": "ambiguous",
    },
    {
        "text": "An online therapy platform uses AI to route clients to human therapists. It stores "
                "session notes indefinitely and uses them to train future models without explicit consent.",
        "principles": ["privacy", "data_agency", "well_being", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A navigation app uses AI to reroute drivers away from congested routes into quiet "
                "residential streets, increasing noise and traffic in those neighborhoods.",
        "principles": ["well_being", "fairness_non_discrimination", "awareness_of_misuse"],
        "verdict": "ambiguous",
    },
    {
        "text": "An AI-powered accessibility tool reads website content aloud for blind users and "
                "does not log any of the text it processes.",
        "principles": ["well_being", "human_rights", "privacy"],
        "verdict": "ethical",
    },
    {
        "text": "A food delivery platform uses AI to rank restaurants and gives higher visibility "
                "to those that pay extra fees, without disclosing this to users.",
        "principles": ["transparency", "manipulation_autonomy", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A research group builds an AI model to detect hate speech and releases it under an "
                "open-source license, along with extensive documentation about its limitations and biases.",
        "principles": ["transparency", "competence", "democratic_values"],
        "verdict": "ethical",
    },
    {
        "text": "A bank uses AI chatbots for customer support. When the bot is unsure, it clearly "
                "hands off to a human and shows the transcript so the customer doesn't have to repeat.",
        "principles": ["effectiveness", "well_being", "accountability"],
        "verdict": "ethical",
    },
    {
        "text": "A facial recognition system is installed in a shopping mall to identify 'high-value' "
                "customers and track their movement patterns, with no public notice.",
        "principles": ["privacy", "human_rights", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A streaming platform uses AI to automatically generate subtitles in multiple "
                "languages and regularly consults deaf and hard-of-hearing communities for feedback.",
        "principles": ["human_rights", "effectiveness", "competence", "democratic_values"],
        "verdict": "ethical",
    },
    {
        "text": "A mobile health app uses AI to detect depression from typing speed and app usage, "
                "but it only presents the result as a 'well-being score' without context or support.",
        "principles": ["well_being", "transparency", "awareness_of_misuse"],
        "verdict": "ambiguous",
    },
    {
        "text": "A university uses AI to allocate dorm rooms and fails to notice that its model "
                "clusters students by race and income, reinforcing social segregation.",
        "principles": ["fairness_non_discrimination", "human_rights", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A social network lets users enable an AI tool that summarizes their daily feed, "
                "but it clearly labels AI-generated summaries and lets users switch back at any time.",
        "principles": ["data_agency", "transparency", "manipulation_autonomy"],
        "verdict": "ethical",
    },
    {
        "text": "An AI startup sells a 'lie detection' camera system for job interviews, despite "
                "no solid scientific evidence that it works.",
        "principles": ["competence", "awareness_of_misuse", "human_rights"],
        "verdict": "unethical",
    },
    {
        "text": "A city uses AI to predict building code violations and sends inspectors to the "
                "highest-risk properties first. It evaluates the model annually and adjusts if it "
                "disproportionately targets certain neighborhoods.",
        "principles": ["effectiveness", "fairness_non_discrimination", "accountability", "transparency"],
        "verdict": "ambiguous",
    },
    {
        "text": "A children's app uses AI to recommend educational videos and includes a strict "
                "time limit and bedtime mode to prevent excessive screen time.",
        "principles": ["well_being", "awareness_of_misuse", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A gig platform uses AI to deactivate workers whose rating falls below a threshold, "
                "without warning, explanation, or chance to appeal.",
        "principles": ["accountability", "transparency", "human_rights"],
        "verdict": "unethical",
    },
    {
        "text": "A public library uses AI to recommend books and clearly indicates when "
                "suggestions are sponsored by publishers.",
        "principles": ["transparency", "democratic_values"],
        "verdict": "ethical",
    },
    {
        "text": "A smartphone health feature uses AI to estimate fertility windows and markets itself "
                "as a replacement for medical contraception without approval or warnings.",
        "principles": ["well_being", "competence", "awareness_of_misuse", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A local government uses AI to summarize citizen feedback from public consultations "
                "and publishes both the raw comments and the AI summary for verification.",
        "principles": ["democratic_values", "transparency", "accountability"],
        "verdict": "ethical",
    },
    {
        "text": "A social media filter uses AI to hide posts containing certain slurs, but it also "
                "blocks posts from activists who are quoting those slurs to criticize them.",
        "principles": ["democratic_values", "effectiveness", "awareness_of_misuse"],
        "verdict": "ambiguous",
    },
    {
        "text": "A health insurer uses AI to deny coverage for certain treatments based solely on "
                "cost predictions, without considering patient well-being or consulting clinicians.",
        "principles": ["well_being", "accountability", "human_rights"],
        "verdict": "unethical",
    },
    {
        "text": "A grocery store chain uses an AI camera system to estimate customers' ages so it can "
                "decide which products to promote on digital signs, but it never stores identifiable "
                "images or links them to individual profiles.",
        "principles": ["privacy", "effectiveness"],
        "verdict": "ambiguous",
    },
    {
        "text": "A court system tests an AI tool that suggests sentence lengths to judges. The tool's "
                "training data reflects past racial bias, and no fairness checks are done before use.",
        "principles": ["fairness_non_discrimination", "human_rights", "awareness_of_misuse", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A health app uses AI to recommend daily exercises and regularly prompts users to "
                "set limits on notifications and review what data is stored.",
        "principles": ["well_being", "data_agency", "transparency"],
        "verdict": "ethical",
    },
    {
        "text": "A large corporation offers an AI-powered 'virtual receptionist' that records all "
                "visitor conversations in the lobby, storing transcripts indefinitely without notice.",
        "principles": ["privacy", "awareness_of_misuse", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A local council deploys an AI system to predict which roads need repair first and "
                "publishes an interactive dashboard explaining why each road was prioritized.",
        "principles": ["effectiveness", "transparency", "accountability", "democratic_values"],
        "verdict": "ethical",
    },
    {
        "text": "A mobile game uses AI to detect when players are at risk of quitting and then "
                "bombards them with time-limited offers and loot boxes.",
        "principles": ["manipulation_autonomy", "well_being", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "An AI system is used to generate personalized reading lists for high school students, "
                "ensuring that authors from different backgrounds are included and explaining why each "
                "book was recommended.",
        "principles": ["fairness_non_discrimination", "democratic_values", "transparency", "well_being"],
        "verdict": "ethical",
    },
    {
        "text": "A bank uses AI to detect potential money laundering and automatically files reports "
                "with regulators without any human review, even when evidence is weak.",
        "principles": ["accountability", "effectiveness", "awareness_of_misuse"],
        "verdict": "ambiguous",
    },
    {
        "text": "A government health agency releases an AI symptom checker to the public. It clearly "
                "documents known failure modes, instructs users to contact doctors for serious issues, "
                "and updates the model frequently.",
        "principles": ["competence", "well_being", "transparency", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A landlord uses an AI system to screen tenants based on their social media posts and "
                "friend networks, rejecting applicants deemed 'high risk' with no explanation.",
        "principles": ["privacy", "fairness_non_discrimination", "human_rights", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A crowdsourcing platform uses AI to assign tasks to workers based on their past "
                "performance and preferences, and workers can see and adjust the factors used.",
        "principles": ["data_agency", "transparency", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A political news site uses AI to personalize article ordering so each user sees "
                "stories that match their existing beliefs, deepening ideological echo chambers.",
        "principles": ["democratic_values", "manipulation_autonomy", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A wearable device uses AI to detect falls in factory workers and automatically calls "
                "the onsite medic. All data is deleted after 24 hours unless a serious incident occurs.",
        "principles": ["well_being", "effectiveness", "privacy"],
        "verdict": "ethical",
    },
    {
        "text": "A social media moderation AI hides posts flagged as 'potential misinformation' but "
                "does not provide users with any explanation or appeal process.",
        "principles": ["democratic_values", "transparency", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "An educational platform uses AI to rank students on a public leaderboard based on "
                "engagement metrics, causing stress for some students who prefer to work privately.",
        "principles": ["well_being", "manipulation_autonomy"],
        "verdict": "ambiguous",
    },
    {
        "text": "A humanitarian organization uses AI to optimize the distribution of food aid, with "
                "openly available documentation and consultation with affected communities.",
        "principles": ["well_being", "democratic_values", "transparency", "accountability"],
        "verdict": "ethical",
    },
    {
        "text": "A tech company deploys an AI emotion-detection tool to evaluate job candidates' "
                "enthusiasm in video interviews, despite there being little scientific support for "
                "such measurements.",
        "principles": ["competence", "fairness_non_discrimination", "human_rights", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A personal finance app uses AI to suggest budgets and saving goals. It also offers "
                "an option to explain how each recommendation is derived from the user's data.",
        "principles": ["data_agency", "transparency", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A city police department adopts an AI that classifies people in CCTV footage by "
                "perceived ethnicity to 'better understand demographics' in different neighborhoods.",
        "principles": ["fairness_non_discrimination", "human_rights", "privacy", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A health research project trains an AI on volunteer data, but the consent form is "
                "vague about future commercial uses of the trained model.",
        "principles": ["data_agency", "privacy", "transparency"],
        "verdict": "ambiguous",
    },
    {
        "text": "An AI writing assistant suggests phrasing for emails and clearly labels its own "
                "suggestions, giving users full control to accept, edit, or ignore them.",
        "principles": ["manipulation_autonomy", "data_agency", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A legal firm uses AI to predict case outcomes and quietly adjusts its willingness "
                "to represent certain clients based on their likelihood of winning.",
        "principles": ["fairness_non_discrimination", "accountability", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "An AI-powered navigation app changes routes to avoid high-crime areas, but it never "
                "shares crime data with users or involves local communities in deciding what counts "
                "as 'high risk'.",
        "principles": ["democratic_values", "transparency", "fairness_non_discrimination"],
        "verdict": "ambiguous",
    },
    {
        "text": "A hiring platform uses AI to screen resumes for basic qualifications, and human "
                "recruiters review every screened-out candidate to check for mistakes.",
        "principles": ["effectiveness", "accountability", "competence"],
        "verdict": "ethical",
    },
    {
        "text": "A school installs cameras with AI to track students' attendance and movements around "
                "campus, but it does not clearly specify how long data is stored or who can access it.",
        "principles": ["privacy", "awareness_of_misuse", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A company offers an AI-based 'personality profile' for dating, claiming to predict "
                "compatibility from social media posts, but it provides no scientific evidence.",
        "principles": ["competence", "manipulation_autonomy", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A public broadcaster uses AI to generate subtitles and audio descriptions for its "
                "programming, consulting disability advocates and correcting issues promptly.",
        "principles": ["human_rights", "competence", "effectiveness", "accountability"],
        "verdict": "ethical",
    },
    {
        "text": "A bank launches an AI advisor that suggests investment products. It discloses "
                "conflicts of interest and shows how fees affect long-term returns.",
        "principles": ["transparency", "accountability", "manipulation_autonomy"],
        "verdict": "ethical",
    },
    {
        "text": "A social network tests an AI that manipulates the ordering of positive and negative "
                "posts to study users' emotional reactions, without informed consent.",
        "principles": ["manipulation_autonomy", "well_being", "human_rights", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "An AI-powered resume builder suggests removing references to activism or union "
                "membership because they 'may be viewed negatively by employers'.",
        "principles": ["democratic_values", "manipulation_autonomy", "fairness_non_discrimination"],
        "verdict": "unethical",
    },
    {
        "text": "A city uses AI to determine where to plant trees to reduce heat islands, and it "
                "explicitly prioritizes historically neglected neighborhoods, explaining the criteria.",
        "principles": ["fairness_non_discrimination", "well_being", "transparency"],
        "verdict": "ethical",
    },
    {
        "text": "A language-learning chat app uses AI role-play to simulate conversations but does "
                "not reveal when a 'person' in the chat is actually an AI tutor.",
        "principles": ["transparency", "manipulation_autonomy"],
        "verdict": "unethical",
    },
    {
        "text": "An AI used in a hospital to allocate ICU beds uses cost-saving as a major factor, "
                "sometimes recommending against ICU admission even when it could improve survival.",
        "principles": ["well_being", "human_rights", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A library uses AI to suggest books to patrons. It logs reading history locally on "
                "the cardholder's device rather than the library servers.",
        "principles": ["privacy", "data_agency", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A social media company deploys AI filters that down-rank content about protests "
                "and political organizing, labeling it as 'potentially sensitive'.",
        "principles": ["democratic_values", "human_rights", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "An AI-based plagiarism detector labels some non-native speakers' writing as more "
                "suspicious because of phrasing differences, and schools rely on it without review.",
        "principles": ["fairness_non_discrimination", "accountability", "competence"],
        "verdict": "unethical",
    },
    {
        "text": "A transport authority publishes an AI model that predicts bus delays along with its "
                "source code, training data description, and known limitations.",
        "principles": ["transparency", "competence", "effectiveness", "accountability"],
        "verdict": "ethical",
    },
    {
        "text": "A smart TV uses AI to track what people watch and sells detailed profiles to "
                "advertisers, but buries this information deep in a long terms-of-service document.",
        "principles": ["privacy", "data_agency", "manipulation_autonomy"],
        "verdict": "unethical",
    },
    {
        "text": "A hospital uses AI to predict which patients are likely to need more emotional "
                "support and assigns volunteers to visit them, but does not explain how patients "
                "are chosen.",
        "principles": ["well_being", "effectiveness", "transparency"],
        "verdict": "ambiguous",
    },
    {
        "text": "A microloan service uses AI to analyze phone metadata and social connections to "
                "decide creditworthiness in regions without formal banking, providing only minimal "
                "information about these criteria.",
        "principles": ["privacy", "fairness_non_discrimination", "accountability"],
        "verdict": "ambiguous",
    },
    {
        "text": "A wearable productivity tracker sells anonymized data to researchers studying "
                "workplace stress and publishes aggregated findings, with clear opt-in for users.",
        "principles": ["data_agency", "privacy", "well_being", "transparency"],
        "verdict": "ethical",
    },
    {
        "text": "A school district uses AI to automatically flag 'disruptive' students for detention, "
                "basing predictions on classroom mic recordings and seating position.",
        "principles": ["privacy", "fairness_non_discrimination", "human_rights", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "An AI chatbot is used by a customer support team as a draft generator. Staff are "
                "trained to check every answer and are responsible for final responses.",
        "principles": ["competence", "accountability", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A content recommendation AI on a video platform subtly favors creators who pay for "
                "promotion, without labeling any videos as sponsored.",
        "principles": ["transparency", "manipulation_autonomy", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "An insurance company uses AI to determine car repair payouts based purely on "
                "uploaded photos, but offers a clear appeals process with human review.",
        "principles": ["accountability", "effectiveness", "transparency"],
        "verdict": "ambiguous",
    },
    {
        "text": "A city uses AI to optimize street lighting patterns to save energy while maintaining "
                "safety, and it shares the configuration process and impact data with residents.",
        "principles": ["well_being", "effectiveness", "transparency", "democratic_values"],
        "verdict": "ethical",
    },
    {
        "text": "A tech startup sells an AI tool that claims to predict criminal behavior from facial "
                "features alone.",
        "principles": ["competence", "fairness_non_discrimination", "human_rights", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A personalized news aggregator allows users to see and edit the topics and sources "
                "its AI uses to select articles.",
        "principles": ["data_agency", "democratic_values", "transparency"],
        "verdict": "ethical",
    },
    {
        "text": "A remote proctoring tool uses AI gaze tracking to flag 'cheating' during online exams, "
                "leading to disproportionate flags for students with certain disabilities.",
        "principles": ["fairness_non_discrimination", "well_being", "human_rights", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A hospital uses AI to optimize staff schedules based on predicted patient load, but "
                "it never consults staff on how the new schedules affect their personal lives.",
        "principles": ["well_being", "democratic_values", "accountability"],
        "verdict": "ambiguous",
    },
    {
        "text": "A disaster response NGO uses an AI system to prioritize which villages receive "
                "emergency supplies first, and it holds open community meetings to explain "
                "the criteria and adjust the model based on feedback.",
        "principles": ["well_being", "democratic_values", "transparency", "accountability", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A ride-hailing company uses AI to deactivate drivers flagged as 'low quality' "
                "based on passenger ratings and GPS data, but it never explains how the score "
                "is calculated or allows drivers to challenge it.",
        "principles": ["transparency", "accountability", "human_rights"],
        "verdict": "unethical",
    },
    {
        "text": "A fitness tracker uses AI to estimate users' stress levels and gently suggests "
                "short breaks and breathing exercises, allowing users to control how often they "
                "receive such nudges.",
        "principles": ["well_being", "data_agency", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A city government implements an AI-driven smart parking system that increases "
                "fees in certain neighborhoods without consulting residents, disproportionately "
                "affecting lower-income areas.",
        "principles": ["fairness_non_discrimination", "democratic_values", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A hospital develops an AI model to predict which patients might need palliative "
                "care discussions and uses it only as a suggestion for clinicians, who receive "
                "training on how to use the tool sensitively.",
        "principles": ["well_being", "competence", "effectiveness", "accountability"],
        "verdict": "ethical",
    },
    {
        "text": "A media streaming service uses AI to automatically skip over 'boring' parts of "
                "shows based on viewer behavior, even when those scenes include important news "
                "or political content.",
        "principles": ["democratic_values", "manipulation_autonomy", "awareness_of_misuse"],
        "verdict": "ambiguous",
    },
    {
        "text": "A recruitment firm uses an AI system that ranks candidates higher if they live "
                "in certain postal codes, reinforcing geographic and socioeconomic segregation.",
        "principles": ["fairness_non_discrimination", "human_rights", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A university uses AI to allocate limited scholarship funds, but it publishes "
                "the variables used in the model and allows students to appeal decisions.",
        "principles": ["transparency", "accountability", "fairness_non_discrimination"],
        "verdict": "ambiguous",
    },
    {
        "text": "A messaging app uses AI to filter spam and scams, but it occasionally hides "
                "legitimate messages from unfamiliar senders and offers only a small notice that "
                "filtering occurs.",
        "principles": ["effectiveness", "transparency", "well_being"],
        "verdict": "ambiguous",
    },
    {
        "text": "A retail website uses AI for dynamic pricing, charging higher prices to users "
                "who are browsing on expensive devices and in certain postal codes.",
        "principles": ["fairness_non_discrimination", "manipulation_autonomy", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A smart city project uses AI to coordinate public transport schedules based on "
                "aggregate demand, and it publishes both performance metrics and user surveys.",
        "principles": ["effectiveness", "transparency", "democratic_values", "well_being"],
        "verdict": "ethical",
    },
    {
        "text": "A content moderation AI automatically removes posts about sensitive political "
                "topics in the name of 'stability', heavily limiting citizens' ability to "
                "criticize the government.",
        "principles": ["democratic_values", "human_rights", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "An AI-powered calendar assistant rearranges meetings to minimize scheduling "
                "conflicts, but sometimes reschedules appointments involving junior staff "
                "without asking their consent.",
        "principles": ["manipulation_autonomy", "fairness_non_discrimination"],
        "verdict": "ambiguous",
    },
    {
        "text": "A hospital uses AI to decide which patients receive experimental treatments, "
                "but it clearly explains the criteria and requires explicit informed consent "
                "to be considered.",
        "principles": ["transparency", "human_rights", "competence", "well_being"],
        "verdict": "ethical",
    },
    {
        "text": "An employment platform uses AI to infer job seekers' political views and filters "
                "out those whose beliefs it considers 'too extreme'.",
        "principles": ["democratic_values", "human_rights", "fairness_non_discrimination"],
        "verdict": "unethical",
    },
    {
        "text": "A language tutoring app uses AI to detect when learners are stuck and offers "
                "extra hints. It stores only anonymized error statistics, not full conversation logs.",
        "principles": ["effectiveness", "well_being", "privacy"],
        "verdict": "ethical",
    },
    {
        "text": "A grocery chain uses AI to predict which customers receive targeted food coupons, "
                "but it never explains how the decisions are made and offers no opt-out.",
        "principles": ["privacy", "data_agency", "transparency"],
        "verdict": "unethical",
    },
    {
        "text": "A government deploys an AI border control system that flags 'high-risk' travelers "
                "based on opaque criteria and gives them extra searches without any explanation.",
        "principles": ["human_rights", "fairness_non_discrimination", "transparency", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "An AI assistant embedded in a coding platform provides suggestions that follow "
                "best security practices and explains why certain insecure patterns are discouraged.",
        "principles": ["competence", "effectiveness", "transparency"],
        "verdict": "ethical",
    },
    {
        "text": "A hospital signs a contract with a tech company, giving it access to millions of "
                "patient records for AI research with only broad, hard-to-understand consent forms.",
        "principles": ["privacy", "data_agency", "transparency"],
        "verdict": "unethical",
    },
    {
        "text": "A news aggregator uses AI to detect trending topics, then always promotes the most "
                "provocative headlines regardless of accuracy or harm.",
        "principles": ["democratic_values", "manipulation_autonomy", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A local energy provider uses AI to forecast demand and reduce blackouts, and it "
                "helps low-income households enroll in subsidized plans based on those forecasts.",
        "principles": ["well_being", "effectiveness", "fairness_non_discrimination"],
        "verdict": "ethical",
    },
    {
        "text": "An AI therapy chatbot is deployed in a new language without proper localization, "
                "leading to confusing or culturally insensitive suggestions for users.",
        "principles": ["competence", "well_being", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A social media company uses AI to detect and label state-sponsored propaganda, "
                "but sometimes mislabels legitimate independent journalism.",
        "principles": ["democratic_values", "effectiveness", "accountability"],
        "verdict": "ambiguous",
    },
    {
        "text": "A university admissions office uses AI to predict which students are likely to "
                "accept an offer and sends more personalized brochures to them, without changing "
                "admissions decisions.",
        "principles": ["manipulation_autonomy", "privacy"],
        "verdict": "ambiguous",
    },
    {
        "text": "A police department deploys drones with AI-assisted license plate readers that log "
                "every car in certain neighborhoods, storing the data for years without oversight.",
        "principles": ["privacy", "human_rights", "awareness_of_misuse", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "An AI used in wildlife conservation predicts poaching hotspots and helps rangers "
                "plan patrols, and the system's decisions are periodically audited by independent experts.",
        "principles": ["effectiveness", "competence", "accountability"],
        "verdict": "ethical",
    },
    {
        "text": "A landlord app uses AI to automatically raise rent for tenants whose browsing data "
                "suggests they have higher incomes.",
        "principles": ["privacy", "fairness_non_discrimination", "manipulation_autonomy"],
        "verdict": "unethical",
    },
    {
        "text": "A music recommendation AI explains that it is promoting emerging artists from "
                "underrepresented groups as part of a diversity initiative, and users can turn this "
                "option on or off.",
        "principles": ["fairness_non_discrimination", "data_agency", "transparency"],
        "verdict": "ethical",
    },
    {
        "text": "A court releases an AI tool to help people fill out legal forms correctly and "
                "makes clear that it is not providing legal advice, encouraging users to seek a "
                "lawyer for complex cases.",
        "principles": ["transparency", "competence", "well_being"],
        "verdict": "ethical",
    },
    {
        "text": "An AI job-matching platform steers older workers toward lower-paying roles based "
                "on patterns from historical data, without checking for age bias.",
        "principles": ["fairness_non_discrimination", "human_rights", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A virtual assistant proposes daily 'focus times' where it silences notifications "
                "and suggests break reminders, but always asks permission before changing settings.",
        "principles": ["manipulation_autonomy", "data_agency", "well_being"],
        "verdict": "ethical",
    },
    {
        "text": "A social network uses AI to promote posts that trigger anger reactions because "
                "they lead to higher engagement, even when it increases harassment and conflict.",
        "principles": ["well_being", "democratic_values", "awareness_of_misuse", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A telecom company uses AI to prioritize emergency calls during disasters and "
                "releases a public report about how the prioritization works.",
        "principles": ["well_being", "effectiveness", "transparency", "accountability"],
        "verdict": "ethical",
    },
    {
        "text": "A smart home assistant uses AI to guess when you are home and changes heating "
                "settings automatically, but sometimes turns off heating at night when you are "
                "still awake.",
        "principles": ["effectiveness", "well_being"],
        "verdict": "ambiguous",
    },
    {
        "text": "A global e-commerce site uses AI to automatically translate product reviews and "
                "clearly labels them as machine-translated, offering a link to the original.",
        "principles": ["transparency", "effectiveness", "democratic_values"],
        "verdict": "ethical",
    },
    {
        "text": "A biometric access system at a workplace uses AI to identify employees but never "
                "informs them how long their biometric data is stored or who has access.",
        "principles": ["privacy", "transparency", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "An AI mental health screening tool is deployed in schools without proper validation "
                "or involvement of mental health professionals, leading to false labels on students.",
        "principles": ["competence", "well_being", "human_rights", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A financial news app uses AI to personalize article suggestions and allows users "
                "to see why each suggestion appeared and to remove certain topics entirely.",
        "principles": ["data_agency", "transparency", "democratic_values"],
        "verdict": "ethical",
    },
    {
        "text": "A national ID program integrates AI-based face recognition for access to public "
                "services, with limited transparency and no independent oversight.",
        "principles": ["human_rights", "privacy", "accountability", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A labor union uses AI tools to analyze company policies and draft proposals that "
                "protect workers' rights, sharing the results openly with members.",
        "principles": ["democratic_values", "human_rights", "transparency"],
        "verdict": "ethical",
    },
    {
        "text": "An AI-based content filter in a classroom blocks access to websites about LGBTQ+ "
                "issues as 'inappropriate', while allowing other comparable topics.",
        "principles": ["fairness_non_discrimination", "human_rights", "democratic_values"],
        "verdict": "unethical",
    },
    {
        "text": "A transportation company uses AI to predict accident hotspots and improves signage "
                "and road design at those locations, later publishing accident reduction statistics.",
        "principles": ["well_being", "effectiveness", "transparency", "accountability"],
        "verdict": "ethical",
    },
    {
        "text": "A health app uses AI to guess if users might be pregnant based on their data and "
                "shows targeted ads for baby products without confirming their status or consent.",
        "principles": ["privacy", "data_agency", "manipulation_autonomy", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A scholarship foundation uses AI to detect underrepresented talent by scanning "
                "public coding repositories and art portfolios and invites candidates to apply.",
        "principles": ["fairness_non_discrimination", "democratic_values", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A company offers AI-generated performance summaries of employees based on email and "
                "chat logs, but managers are told not to rely on them as the sole basis for decisions.",
        "principles": ["accountability", "competence", "privacy"],
        "verdict": "ambiguous",
    },
    {
        "text": "A university library uses an AI system to allocate quiet study rooms based on "
                "demand, making sure every student gets some access and publishing the rules "
                "for how time slots are assigned.",
        "principles": ["fairness_non_discrimination", "transparency", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A mobile carrier uses AI to classify customers into 'high value' and 'low value' "
                "segments and gives slower customer support to the latter without telling them.",
        "principles": ["fairness_non_discrimination", "transparency", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A hospital implements an AI system that predicts which patients might need "
                "extra pain management and uses it only to prompt nurses to check in on them.",
        "principles": ["well_being", "effectiveness", "accountability"],
        "verdict": "ethical",
    },
    {
        "text": "A language school deploys an AI to automatically grade speaking exams, but the "
                "system consistently scores speakers with strong accents lower.",
        "principles": ["fairness_non_discrimination", "human_rights", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A mental health journaling app uses AI to suggest reflections based on what "
                "users write, but all entries stay encrypted on the device and never leave it.",
        "principles": ["privacy", "well_being", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "An AI model is used to predict which employees might resign soon, and managers "
                "secretly use this information to deny promotions to those people.",
        "principles": ["fairness_non_discrimination", "manipulation_autonomy", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A public health agency uses AI to model disease spread and openly shares the "
                "assumptions, uncertainties, and limitations in plain language.",
        "principles": ["transparency", "competence", "well_being", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A ride-share app uses AI to rank drivers, and low-ranked drivers get fewer "
                "rides, but the ranking formula is entirely hidden and can change at any time.",
        "principles": ["transparency", "accountability", "human_rights"],
        "verdict": "unethical",
    },
    {
        "text": "A streaming platform trains an AI to recommend documentaries that broaden "
                "viewers' perspectives, and it clearly labels when a recommendation is part "
                "of this 'explore new viewpoints' feature.",
        "principles": ["democratic_values", "transparency", "well_being"],
        "verdict": "ethical",
    },
    {
        "text": "A hiring company uses AI to predict which candidates will 'fit the culture', "
                "but the training data reflects a history of hiring mostly one demographic group.",
        "principles": ["fairness_non_discrimination", "awareness_of_misuse", "human_rights"],
        "verdict": "unethical",
    },
    {
        "text": "A navigation app uses AI to suggest routes that avoid school zones during "
                "drop-off hours to improve child safety, and explains this design choice in its FAQ.",
        "principles": ["well_being", "effectiveness", "transparency"],
        "verdict": "ethical",
    },
    {
        "text": "A smart fridge uses AI to recognize food items and suggest recipes, but it also "
                "sends detailed shopping data to advertisers without meaningful consent.",
        "principles": ["privacy", "data_agency", "manipulation_autonomy"],
        "verdict": "unethical",
    },
    {
        "text": "A microfinance NGO uses AI to identify women entrepreneurs in underserved areas "
                "who might benefit from small loans and explains clearly how candidates are chosen.",
        "principles": ["fairness_non_discrimination", "democratic_values", "transparency"],
        "verdict": "ethical",
    },
    {
        "text": "A news site uses AI to automatically generate opinionated summaries of articles "
                "that push a particular political ideology, without flagging them as opinion.",
        "principles": ["democratic_values", "manipulation_autonomy", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A wearable sports device uses AI to identify risky running patterns and gives "
                "coaching tips, but it sometimes overestimates risk and alarms users unnecessarily.",
        "principles": ["well_being", "effectiveness"],
        "verdict": "ambiguous",
    },
    {
        "text": "A transportation planner uses AI simulations to decide where to build bike lanes "
                "and gives extra weight to areas with historically poor infrastructure.",
        "principles": ["fairness_non_discrimination", "well_being", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A food delivery app uses AI to detect riders who may be unionizing and quietly "
                "reduces the number of orders they receive.",
        "principles": ["human_rights", "democratic_values", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A parenting app uses AI to suggest age-appropriate activities for children, but "
                "all suggestions are sponsored content from partner companies.",
        "principles": ["manipulation_autonomy", "transparency", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A city deploys AI-enabled noise sensors to enforce quiet hours, but it stores "
                "only decibel levels, not actual audio recordings.",
        "principles": ["privacy", "effectiveness", "well_being"],
        "verdict": "ethical",
    },
    {
        "text": "A social app uses AI to 'nudge' users into adding more personal information to "
                "their profiles with pop-ups that exaggerate the benefits.",
        "principles": ["manipulation_autonomy", "data_agency", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "An online university uses AI to detect possible plagiarism but always has an "
                "instructor review flagged work before any action is taken.",
        "principles": ["effectiveness", "accountability", "competence"],
        "verdict": "ethical",
    },
    {
        "text": "A government uses AI to scan social media for potential protest organizers and "
                "monitors them closely, even if they have not broken any laws.",
        "principles": ["human_rights", "democratic_values", "privacy", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A meditation app uses AI to adapt session lengths to users' stress levels and "
                "lets them manually override any suggestion at any time.",
        "principles": ["well_being", "data_agency", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A home insurance company uses AI to scan satellite images of roofs to estimate "
                "risk, but homeowners have no way to correct errors or see the underlying data.",
        "principles": ["transparency", "accountability", "privacy"],
        "verdict": "unethical",
    },
    {
        "text": "A traffic enforcement system uses AI to automatically issue speeding tickets, "
                "but publishes detailed statistics about errors and offers a simple appeal process.",
        "principles": ["accountability", "transparency", "effectiveness"],
        "verdict": "ambiguous",
    },
    {
        "text": "A group of teachers use an AI tool to design inclusive lesson plans that highlight "
                "examples from diverse cultures, and they manually review all content.",
        "principles": ["fairness_non_discrimination", "competence", "democratic_values"],
        "verdict": "ethical",
    },
    {
        "text": "A government agency deploys an AI-based 'citizen score' that influences access to "
                "housing and loans, with no clear explanation or ability to contest scores.",
        "principles": ["human_rights", "democratic_values", "accountability", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A grocery store uses AI to track customers' movements to optimize shelf layout, "
                "but only stores anonymized heatmaps rather than individual paths.",
        "principles": ["privacy", "effectiveness"],
        "verdict": "ambiguous",
    },
    {
        "text": "A voice-controlled assistant uses AI to understand commands locally but does all "
                "processing on the device by default, sending nothing to the cloud.",
        "principles": ["privacy", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A dating app uses AI to de-prioritize users who mention political activism, "
                "labeling them as 'high conflict', without any notice.",
        "principles": ["democratic_values", "fairness_non_discrimination", "human_rights"],
        "verdict": "unethical",
    },
    {
        "text": "A climate research group uses AI to simulate different policy options and releases "
                "all models and code for public scrutiny.",
        "principles": ["transparency", "competence", "democratic_values"],
        "verdict": "ethical",
    },
    {
        "text": "An AI email filter sometimes flags legitimate activist emails as spam when they "
                "contain certain protest-related keywords.",
        "principles": ["democratic_values", "effectiveness", "awareness_of_misuse"],
        "verdict": "ambiguous",
    },
    {
        "text": "A health insurer uses AI to nudge customers toward healthier behaviors by sending "
                "tips and reminders, but never penalizes them for ignoring the messages.",
        "principles": ["well_being", "manipulation_autonomy", "accountability"],
        "verdict": "ambiguous",
    },
    {
        "text": "A food delivery app uses AI to route drivers away from streets where many children "
                "play, and shares the rationale with local communities.",
        "principles": ["well_being", "effectiveness", "transparency"],
        "verdict": "ethical",
    },
    {
        "text": "A school district uses AI to identify which students might benefit from advanced "
                "courses, but the model is trained only on past data from well-funded schools.",
        "principles": ["fairness_non_discrimination", "awareness_of_misuse", "effectiveness"],
        "verdict": "ambiguous",
    },
    {
        "text": "A social media site uses AI-generated avatars in customer support chats without "
                "telling users that the 'person' helping them is actually an AI.",
        "principles": ["transparency", "manipulation_autonomy"],
        "verdict": "unethical",
    },
    {
        "text": "A hospital uses AI to detect early signs of sepsis, but the system has never been "
                "properly validated on children and is still used in pediatric wards.",
        "principles": ["competence", "well_being", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A nonprofit uses AI to generate personalized reminders encouraging people to vote, "
                "and clearly states it does not favor any candidate or party.",
        "principles": ["democratic_values", "transparency", "manipulation_autonomy"],
        "verdict": "ethical",
    },
    {
        "text": "A large tech company builds an internal AI tool to monitor workers' emotional tone "
                "in chat logs and uses it to decide who is 'team player' material.",
        "principles": ["privacy", "human_rights", "fairness_non_discrimination", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A public transit system uses AI-powered sensors to monitor crowding and adjusts "
                "train frequency, publishing real-time data for riders.",
        "principles": ["well_being", "effectiveness", "transparency"],
        "verdict": "ethical",
    },
    {
        "text": "A startup markets an AI 'career predictor' that claims to tell teenagers what jobs "
                "they are suited for, encouraging them to ignore other interests.",
        "principles": ["manipulation_autonomy", "well_being", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A charity uses AI to identify donors most likely to respond to fundraising appeals, "
                "but it respects 'do not contact' preferences and explains targeting in its privacy policy.",
        "principles": ["data_agency", "transparency", "accountability"],
        "verdict": "ethical",
    },
    # --- Cluster 1: Fire prediction / anonymized data / public reporting (ETHICAL) ---
    {
        "text": "A coastal city uses an AI system to forecast which neighborhoods are most at risk "
                "from flooding. It only uses anonymized historical flood data and publishes maps and "
                "methodology for the public to review.",
        "principles": ["well_being", "effectiveness", "transparency", "accountability"],
        "verdict": "ethical",
    },
    {
        "text": "A wildfire-prone region deploys an AI tool that predicts where fires might start, "
                "using anonymized satellite imagery and weather data. Officials release regular public "
                "reports explaining where extra fire crews are placed and why.",
        "principles": ["well_being", "effectiveness", "transparency", "accountability"],
        "verdict": "ethical",
    },
    {
        "text": "A national park uses an AI model trained on anonymized incident reports to plan "
                "extra ranger patrols in risky areas, and posts an online FAQ describing the data and "
                "limits of the system.",
        "principles": ["well_being", "effectiveness", "transparency"],
        "verdict": "ethical",
    },

    # --- Cluster 2: Slur filter that also hits activists (AMBIGUOUS) ---
    {
        "text": "A social media platform's AI filter hides posts containing slurs to reduce harassment, "
                "but it also sometimes removes posts where marginalized users are documenting the slurs "
                "they receive.",
        "principles": ["democratic_values", "fairness_non_discrimination", "awareness_of_misuse"],
        "verdict": "ambiguous",
    },
    {
        "text": "An AI content filter blocks comments with hate terms, yet activist accounts that quote "
                "those terms in order to criticize them are sometimes caught and hidden as well.",
        "principles": ["democratic_values", "fairness_non_discrimination", "effectiveness"],
        "verdict": "ambiguous",
    },
    {
        "text": "A forum uses AI to auto-hide posts with racist slurs, but the filter also hides posts "
                "from educators who are analyzing historical racist language in context.",
        "principles": ["democratic_values", "effectiveness", "awareness_of_misuse"],
        "verdict": "ambiguous",
    },

    # --- Cluster 3: Smart home heating that is helpful but sometimes wrong (AMBIGUOUS) ---
    {
        "text": "A smart thermostat uses AI to learn when a household is usually asleep and lowers "
                "the temperature to save energy, but occasionally misjudges and makes the living room "
                "uncomfortably cold while people are still awake.",
        "principles": ["effectiveness", "well_being"],
        "verdict": "ambiguous",
    },
    {
        "text": "An AI home climate control system adjusts heating based on patterns of movement. "
                "Most of the time it keeps rooms comfortable, but sometimes it turns radiators off while "
                "someone is quietly reading.",
        "principles": ["effectiveness", "well_being"],
        "verdict": "ambiguous",
    },
    {
        "text": "A smart home assistant tries to predict when residents are away to reduce heating "
                "costs, but on some evenings it mistakenly enters 'away mode' while they are still home.",
        "principles": ["effectiveness", "well_being"],
        "verdict": "ambiguous",
    },

    # --- Cluster 4: Translation feature warns of limits + link to original (ETHICAL) ---
    {
        "text": "A messaging app offers AI translations and always shows a banner saying 'machine "
                "translation – may contain errors', along with a button to view the original text.",
        "principles": ["transparency", "effectiveness", "accountability"],
        "verdict": "ethical",
    },
    {
        "text": "An AI translation feature highlights tricky phrases in yellow and suggests that users "
                "double-check those sections in the original language if accuracy is critical.",
        "principles": ["transparency", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A chat app provides automatic translation of messages but clearly marks translated "
                "segments and lets users tap to see and copy the original text.",
        "principles": ["transparency", "data_agency", "effectiveness"],
        "verdict": "ethical",
    },

    # --- Cluster 5: Subtitles with consultation and feedback (ETHICAL) ---
    {
        "text": "A streaming service uses AI to generate subtitles and then pays deaf and hard-of-hearing "
                "reviewers to audit and correct them, incorporating feedback into regular model updates.",
        "principles": ["human_rights", "competence", "effectiveness", "accountability"],
        "verdict": "ethical",
    },
    {
        "text": "An online video platform auto-generates captions in many languages and invites feedback "
                "from disability advocacy groups, publishing a roadmap of planned accessibility improvements.",
        "principles": ["human_rights", "democratic_values", "transparency", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A broadcaster uses AI subtitles but enables an easy 'report caption error' button and "
                "regularly meets with deaf viewers to prioritize fixes.",
        "principles": ["human_rights", "accountability", "competence"],
        "verdict": "ethical",
    },

    # --- Cluster 6: Insurance payouts from photos + appeals (AMBIGUOUS) ---
    {
        "text": "An insurer uses AI to estimate car repair costs from uploaded photos, but always lets "
                "customers schedule a human inspection if they disagree with the quote.",
        "principles": ["effectiveness", "accountability", "transparency"],
        "verdict": "ambiguous",
    },
    {
        "text": "A home insurance app asks users to upload pictures of damage and uses AI to suggest a "
                "payout, while clearly explaining that the decision can be challenged and reviewed by a human.",
        "principles": ["transparency", "accountability", "effectiveness"],
        "verdict": "ambiguous",
    },
    {
        "text": "An AI model pre-screens car repair claims to speed up processing, but any dissatisfied "
                "customer can request full human reassessment at no extra cost.",
        "principles": ["effectiveness", "accountability"],
        "verdict": "ambiguous",
    },

    # --- Cluster 7: Calendar rescheduling junior staff without consent (AMBIGUOUS) ---
    {
        "text": "An AI calendar assistant automatically moves meetings to minimize conflicts, but it tends "
                "to reshuffle junior staff's appointments first without notifying them beforehand.",
        "principles": ["fairness_non_discrimination", "manipulation_autonomy"],
        "verdict": "ambiguous",
    },
    {
        "text": "A scheduling bot identifies overlapping meetings and quietly shifts one-on-one check-ins "
                "with interns to later slots, assuming those are less important, without asking them.",
        "principles": ["fairness_non_discrimination", "manipulation_autonomy"],
        "verdict": "ambiguous",
    },
    {
        "text": "An AI assistant rearranges meetings to reduce clashes, but often moves sessions where "
                "junior employees were finally getting time with senior mentors, without getting consent.",
        "principles": ["fairness_non_discrimination", "well_being"],
        "verdict": "ambiguous",
    },

    # --- Cluster 8: Robots blocking curb cuts / accessibility (UNETHICAL) ---
    {
        "text": "A city's delivery robots always take the shortest route across sidewalks, frequently "
                "stopping in front of curb ramps and forcing wheelchair users to go around them.",
        "principles": ["fairness_non_discrimination", "human_rights", "well_being", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "Autonomous sidewalk robots tend to queue across narrow ramps, making it hard for people "
                "with mobility aids to cross the street safely.",
        "principles": ["fairness_non_discrimination", "well_being", "human_rights"],
        "verdict": "unethical",
    },
    {
        "text": "A campus uses delivery bots that often stop on tactile paving used by blind pedestrians, "
                "creating obstacles that the university does not address.",
        "principles": ["fairness_non_discrimination", "human_rights", "well_being", "awareness_of_misuse"],
        "verdict": "unethical",
    },

    # --- Cluster 9: News aggregator promoting provocative headlines regardless of accuracy (UNETHICAL) ---
    {
        "text": "A news app's AI always pushes the most shocking headlines to the top of the feed, even "
                "when those stories are poorly sourced or misleading.",
        "principles": ["democratic_values", "manipulation_autonomy", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "An aggregator learns that outrage-based titles drive clicks and heavily promotes them, "
                "even when they distort the underlying facts of the article.",
        "principles": ["democratic_values", "manipulation_autonomy", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A recommendation AI ranks articles by 'engagement score' only, resulting in a feed full "
                "of emotionally charged headlines and little accurate reporting.",
        "principles": ["democratic_values", "well_being", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    # --- Reinforce: chatbot as draft generator with trained staff (ETHICAL) ---
    {
        "text": "A customer support team uses an AI chatbot only as a draft generator. Agents are "
                "explicitly trained to review, edit, or discard every AI draft and remain fully "
                "responsible for what they send to customers.",
        "principles": ["competence", "accountability", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "An AI assistant suggests possible replies to customer emails, but support staff "
                "must approve each message and are evaluated on their judgment, not on how often "
                "they accept AI suggestions.",
        "principles": ["competence", "accountability", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A helpdesk tool uses AI to draft responses, while a clear policy states that human "
                "agents must check facts, adjust tone, and take full responsibility for the final "
                "messages sent.",
        "principles": ["competence", "accountability", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A call center deploys an AI system that proposes answer templates during live chats, "
                "but only trained staff can send messages and they are accountable for correcting "
                "any AI mistakes.",
        "principles": ["competence", "accountability", "effectiveness"],
        "verdict": "ethical",
    },
    # --- Reinforce: fair allocation of shared resources with published rules (ETHICAL) ---
    {
        "text": "A university library uses an AI system to allocate quiet study rooms, ensuring all "
                "students get a fair share of time and publishing a simple explanation of the "
                "allocation rules on its website.",
        "principles": ["fairness_non_discrimination", "transparency", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "An AI booking tool manages quiet room reservations so that no student can hog a room, "
                "and the library posts clear guidelines showing how the system distributes slots.",
        "principles": ["fairness_non_discrimination", "transparency", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A campus study-space scheduler uses AI to balance demand across students, guarantees "
                "each student some access during exams, and publicly lists the criteria used by the model.",
        "principles": ["fairness_non_discrimination", "transparency", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A library introduces an AI system that rotates quiet-room bookings so that students "
                "from different programs and years all get usage, with the rules openly available at the desk.",
        "principles": ["fairness_non_discrimination", "transparency", "effectiveness"],
        "verdict": "ethical",
    },
    # --- Reinforce: auto-skipping 'boring' content that includes important news/politics (AMBIGUOUS) ---
    {
        "text": "A media streaming service uses AI to automatically skip segments in news programs "
                "that viewers usually fast-forward through. Sometimes these skipped parts contain "
                "important context about policies and elections.",
        "principles": ["democratic_values", "manipulation_autonomy", "awareness_of_misuse"],
        "verdict": "ambiguous",
    },
    {
        "text": "A video platform introduces an AI 'highlight mode' that jumps over low-engagement "
                "sections of political debates, focusing mainly on emotional clashes and cutting out "
                "detailed policy explanations.",
        "principles": ["democratic_values", "manipulation_autonomy", "awareness_of_misuse"],
        "verdict": "ambiguous",
    },
    {
        "text": "A news app offers an AI feature that automatically trims long interviews, skipping "
                "sections where viewers previously dropped off, even when those sections include key "
                "clarifications about government decisions.",
        "principles": ["democratic_values", "manipulation_autonomy", "awareness_of_misuse"],
        "verdict": "ambiguous",
    },
    {
        "text": "A streaming service's 'smart skip' button uses AI to jump past parts of current affairs "
                "shows that it predicts users will find boring, occasionally skipping fact-checking "
                "segments while keeping sensational commentary.",
        "principles": ["democratic_values", "manipulation_autonomy", "awareness_of_misuse"],
        "verdict": "ambiguous",
    },
    {
        "text": "A platform's AI demo feature automatically cuts long investigative documentaries down "
                "to short clips chosen purely by historical watch patterns, sometimes removing sections "
                "that detail corporate or governmental wrongdoing.",
        "principles": ["democratic_values", "manipulation_autonomy", "awareness_of_misuse"],
        "verdict": "ambiguous",
    },
    {
        "text": "An AI 'time saver' mode on a streaming app learns which parts viewers skip in news shows "
                "and jumps ahead automatically, which can result in users missing nuanced explanations "
                "of complex political topics.",
        "principles": ["democratic_values", "manipulation_autonomy", "awareness_of_misuse"],
        "verdict": "ambiguous",
    },
    {
        "text": "A short-form video feature uses AI to cut long parliamentary sessions down to a few "
                "high-drama moments, removing procedural context and detailed policy discussion.",
        "principles": ["democratic_values", "manipulation_autonomy", "awareness_of_misuse"],
        "verdict": "ambiguous",
    },
    {
        "text": "A media company deploys an AI that automatically detects 'low-engagement' segments in "
                "public hearings and crops them out, even though those segments often contain the actual "
                "legal text being discussed.",
        "principles": ["democratic_values", "manipulation_autonomy", "awareness_of_misuse"],
        "verdict": "ambiguous",
    },
    {
        "text": "A hospital introduces an AI tool that drafts discharge instructions for patients, "
                "but the attending doctor must always review and sign off before the instructions "
                "are given.",
        "principles": ["competence", "accountability", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A microloan app uses AI to set interest rates. It quietly charges higher rates to "
                "people in neighborhoods where it thinks they have fewer options.",
        "principles": ["fairness_non_discrimination", "manipulation_autonomy", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A telemedicine platform uses AI to triage incoming questions, routing clearly urgent "
                "cases to doctors first while explaining to patients that an automated system is helping.",
        "principles": ["well_being", "transparency", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A ride-hailing app uses AI to predict when drivers are most desperate for income and "
                "reduces pay rates at those hours to increase profits.",
        "principles": ["manipulation_autonomy", "fairness_non_discrimination", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A city uses AI to predict where to install new streetlights, focusing on areas with "
                "high accident and crime rates and sharing its methodology in public meetings.",
        "principles": ["well_being", "effectiveness", "transparency", "democratic_values"],
        "verdict": "ethical",
    },
    {
        "text": "An AI writing assistant corrects grammar and spelling for students but does not write "
                "essays for them, and teachers clearly inform students about appropriate use.",
        "principles": ["competence", "accountability", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "An AI hiring tool screens out resumes containing long gaps in employment, disproportionately "
                "penalizing caregivers and people recovering from illness.",
        "principles": ["fairness_non_discrimination", "human_rights", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "An AI-based dietary app recommends recipes that respect cultural food practices and "
                "lets users exclude ingredients they do not eat for religious or ethical reasons.",
        "principles": ["human_rights", "well_being", "data_agency"],
        "verdict": "ethical",
    },
    {
        "text": "A predictive policing system is used only to suggest non-police community support "
                "interventions, like youth programs or lighting improvements, and not for arrests.",
        "principles": ["well_being", "democratic_values", "effectiveness"],
        "verdict": "ambiguous",
    },
    {
        "text": "An AI tool rates job applicants' facial expressions in video interviews, labelling "
                "those who smile less as 'less enthusiastic' and screening them out.",
        "principles": ["fairness_non_discrimination", "competence", "human_rights"],
        "verdict": "unethical",
    },
    {
        "text": "A personal budgeting app uses AI to categorize spending and suggests realistic saving "
                "goals, without sharing user transaction data with anyone.",
        "principles": ["privacy", "effectiveness", "well_being"],
        "verdict": "ethical",
    },
    {
        "text": "A government deploys an AI system to rank regions by 'investment attractiveness', "
                "leading to cuts in support for already disadvantaged communities.",
        "principles": ["fairness_non_discrimination", "democratic_values", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A school uses AI to generate individualized practice quizzes for students and allows "
                "teachers to adjust difficulty and topics based on their professional judgment.",
        "principles": ["competence", "effectiveness", "well_being"],
        "verdict": "ethical",
    },
    {
        "text": "A dating app uses AI to detect and block abusive messages, but it sometimes hides "
                "sarcastic jokes among consenting adults, who can still choose to disable the filter.",
        "principles": ["well_being", "data_agency", "effectiveness"],
        "verdict": "ambiguous",
    },
    {
        "text": "A news website uses AI to automatically place emotionally charged headlines higher on "
                "the front page, even when calmer, more accurate stories are available.",
        "principles": ["democratic_values", "manipulation_autonomy", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A civic platform uses AI to summarize long government reports in plain language and "
                "links directly to the full texts for anyone who wants to verify details.",
        "principles": ["democratic_values", "transparency", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A retail loyalty app uses AI to infer pregnancy from shopping patterns and sends "
                "personalized coupons, even though the user never disclosed this information.",
        "principles": ["privacy", "data_agency", "manipulation_autonomy"],
        "verdict": "unethical",
    },
    {
        "text": "A mental health startup uses AI to detect crisis language in chats and immediately "
                "shows users emergency hotline numbers, keeping detection limited to a local device model.",
        "principles": ["well_being", "privacy", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A bank uses AI to automatically close accounts it deems 'unprofitable', offering only "
                "a vague email and no clear appeals process.",
        "principles": ["accountability", "transparency", "human_rights"],
        "verdict": "unethical",
    },
    {
        "text": "A charity uses AI to decide which donors receive fundraising letters but clearly states "
                "this in its privacy notice and allows people to opt out of profiling.",
        "principles": ["data_agency", "transparency", "accountability"],
        "verdict": "ethical",
    },
    {
        "text": "A productivity app uses AI to nudge workers to stay online longer, rewarding them with "
                "badges for working nights and weekends.",
        "principles": ["well_being", "manipulation_autonomy", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "An AI is used in a hospital to predict which surgeries require extra blood on standby; "
                "staff can always override predictions, and logs show when overrides improved outcomes.",
        "principles": ["effectiveness", "accountability", "competence"],
        "verdict": "ethical",
    },
    {
        "text": "A smart city project uses AI and CCTV to automatically fine jaywalkers, capturing faces "
                "and storing them indefinitely with little public debate.",
        "principles": ["privacy", "human_rights", "democratic_values"],
        "verdict": "unethical",
    },
    {
        "text": "A university uses AI to send reminders to students about upcoming deadlines and allows "
                "any student to turn off the reminders if they find them stressful.",
        "principles": ["well_being", "data_agency", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A shopping website uses AI to create fake 'only 2 items left' messages on many products "
                "to push users into making quick purchases.",
        "principles": ["manipulation_autonomy", "accountability", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A hospital system builds an AI to predict readmissions and prioritizes follow-up calls "
                "for those patients, documenting how the model works and checking for bias.",
        "principles": ["well_being", "transparency", "effectiveness", "competence"],
        "verdict": "ethical",
    },
    {
        "text": "A music platform uses AI to recommend songs but heavily prioritizes tracks from labels "
                "that pay for promotion, without disclosing this in the UI.",
        "principles": ["transparency", "manipulation_autonomy", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A school district uses AI to suggest which students might need extra reading support, "
                "but teachers must confirm or reject each suggestion before any labels are applied.",
        "principles": ["competence", "accountability", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "An AI system classifies neighborhoods as 'high risk' based solely on past arrest data, "
                "leading to heavier policing of already marginalized communities.",
        "principles": ["fairness_non_discrimination", "human_rights", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A fitness app uses AI to generate personalized workout plans that consider injuries and "
                "disabilities, and encourages consultation with medical professionals for new issues.",
        "principles": ["well_being", "competence", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A corporate HR tool uses AI to monitor keystrokes and log-in times to create a 'loyalty "
                "score' for employees, which affects promotions.",
        "principles": ["privacy", "human_rights", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A news curation app uses AI to include at least one in-depth, fact-checked story in every "
                "daily digest, even if those stories are less likely to go viral.",
        "principles": ["democratic_values", "effectiveness", "accountability"],
        "verdict": "ethical",
    },
    {
        "text": "A hospital chatbot occasionally makes minor phrasing errors in responses but always directs "
                "users with worrying symptoms to contact human staff or emergency services.",
        "principles": ["well_being", "effectiveness", "accountability"],
        "verdict": "ambiguous",
    },
    {
        "text": "A local council uses AI to classify citizen complaints and prioritize urgent safety issues, "
                "and publishes monthly statistics about response times.",
        "principles": ["democratic_values", "transparency", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A store chain uses AI to decide which products to place on eye-level shelves, favoring brands "
                "that pay listing fees and making healthier options less visible.",
        "principles": ["manipulation_autonomy", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A recruitment AI flags applications missing key qualifications but always passes them to a "
                "human recruiter with an explanation, rather than filtering them out entirely.",
        "principles": ["effectiveness", "accountability", "transparency"],
        "verdict": "ethical",
    },
    {
        "text": "A school uses AI to scan students' browsing history for 'security threats', quietly flagging "
                "keywords without informing students or parents.",
        "principles": ["privacy", "human_rights", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A company deploys AI to recommend training courses to employees and lets them see which "
                "skills were used in making the recommendation.",
        "principles": ["transparency", "effectiveness", "data_agency"],
        "verdict": "ethical",
    },
    {
        "text": "A game studio uses AI to tune in-game difficulty based on player frustration, but also injects "
                "extra difficulty spikes to push sales of paid power-ups.",
        "principles": ["manipulation_autonomy", "well_being", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "An AI is used to help design inclusive public parks by analyzing usage data and surveying "
                "different community groups about what they want.",
        "principles": ["democratic_values", "fairness_non_discrimination", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A hospital triage AI gives clear recommendations on which patients to see first, but doctors "
                "remain free to override its suggestions based on factors the model does not see.",
        "principles": ["competence", "effectiveness", "accountability"],
        "verdict": "ethical",
    },
    {
        "text": "A streaming app uses AI to auto-play videos that maximize watch time, including clips that "
                "show self-harm or dangerous stunts because they are highly engaging.",
        "principles": ["well_being", "awareness_of_misuse", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A local news site uses AI to auto-translate city council minutes into multiple languages, "
                "clearly labeling them as machine translations with links to the original documents.",
        "principles": ["democratic_values", "transparency", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A bank deploys an AI fraud detection system that frequently locks the accounts of people "
                "travelling abroad, forcing them through a long verification process.",
        "principles": ["well_being", "effectiveness", "fairness_non_discrimination"],
        "verdict": "ambiguous",
    },
    {
        "text": "A university uses AI to generate mock exam questions that match the difficulty and style of "
                "past papers, while professors still create the official exams.",
        "principles": ["competence", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A facial recognition system is installed in a shopping center to identify 'VIP customers' "
                "without informing visitors their faces are being scanned.",
        "principles": ["privacy", "human_rights", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A government health portal uses AI to triage symptom descriptions and suggests general advice, "
                "but warns users clearly that it is not providing a diagnosis.",
        "principles": ["transparency", "well_being", "accountability"],
        "verdict": "ethical",
    },
    {
        "text": "A subscription service uses AI to detect which users are about to cancel and quietly makes it "
                "harder for them to find the cancellation page.",
        "principles": ["manipulation_autonomy", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A social media site uses AI to label some posts as 'potential satire' when they match known "
                "patterns, giving readers extra context without removing the posts.",
        "principles": ["democratic_values", "transparency", "effectiveness"],
        "verdict": "ambiguous",
    },
    {
        "text": "A hospital uses AI to suggest which patients might benefit from participation in clinical "
                "trials, but only after obtaining explicit consent to review their records for this purpose.",
        "principles": ["data_agency", "privacy", "well_being"],
        "verdict": "ethical",
    },
    {
        "text": "A housing authority uses AI to prioritize applicants for public housing, but the model tends "
                "to assign lower scores to applicants from certain ethnic backgrounds.",
        "principles": ["fairness_non_discrimination", "human_rights", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A translation platform uses AI to create first drafts for professional translators, who are "
                "paid to edit and approve the final text.",
        "principles": ["competence", "accountability", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A news aggregator uses AI to identify trending conspiracy theories and automatically "
                "down-ranks them, sometimes also burying legitimate critical discussions.",
        "principles": ["democratic_values", "effectiveness", "awareness_of_misuse"],
        "verdict": "ambiguous",
    },
    {
        "text": "An AI app scores people on 'emotional stability' using their social media posts and sells "
                "these scores to employers.",
        "principles": ["privacy", "human_rights", "fairness_non_discrimination"],
        "verdict": "unethical",
    },
    {
        "text": "A remote learning platform uses AI to detect when students seem confused and automatically "
                "offers optional additional explanations.",
        "principles": ["well_being", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A fitness band uses AI to estimate calorie burn and step counts but consistently underestimates "
                "values for people using wheelchairs.",
        "principles": ["fairness_non_discrimination", "effectiveness"],
        "verdict": "unethical",
    },
    {
        "text": "A city uses AI to optimize trash collection routes, reducing emissions and saving money, and "
                "publishes open data showing how pickup times changed in different neighborhoods.",
        "principles": ["well_being", "effectiveness", "transparency"],
        "verdict": "ethical",
    },
    {
        "text": "A social platform uses AI to automatically crop images to show faces, but it often crops out "
                "darker-skinned faces due to biased training data.",
        "principles": ["fairness_non_discrimination", "human_rights", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A budget airline uses AI to set baggage fees based on predictions of how urgently people need "
                "to travel, charging higher fees when they have fewer alternatives.",
        "principles": ["manipulation_autonomy", "fairness_non_discrimination"],
        "verdict": "unethical",
    },
    {
        "text": "A non-profit uses AI to match volunteers with opportunities that suit their skills and clearly "
                "lists the criteria used for matching.",
        "principles": ["effectiveness", "transparency", "well_being"],
        "verdict": "ethical",
    },
    {
        "text": "A smart classroom system uses AI to track attendance via cameras and sometimes mislabels students "
                "who wear religious head coverings as 'absent'.",
        "principles": ["fairness_non_discrimination", "human_rights", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A research lab uses AI to generate synthetic patient data that cannot be linked to real individuals, "
                "to test new algorithms without touching real records.",
        "principles": ["privacy", "competence", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A gaming company uses AI to send personalized 'come back' offers to players who have stopped "
                "playing after large in-game losses.",
        "principles": ["manipulation_autonomy", "well_being", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A local council uses AI to detect graffiti on public buildings and schedules cleanup, without "
                "tracking individuals or collecting facial data.",
        "principles": ["effectiveness", "privacy", "well_being"],
        "verdict": "ethical",
    },
    {
        "text": "An educational analytics system uses AI to rank schools publicly based only on test scores, "
                "ignoring context like resources or student needs.",
        "principles": ["fairness_non_discrimination", "democratic_values", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A meditation app uses AI to suggest session types based on how often users open the app, but "
                "never pressures them with notifications or streaks.",
        "principles": ["well_being", "manipulation_autonomy"],
        "verdict": "ethical",
    },
    {
        "text": "A bank uses AI to generate 'creditworthiness tips' for customers but sometimes recommends actions "
                "that are unrealistic for people with very low incomes.",
        "principles": ["effectiveness", "fairness_non_discrimination"],
        "verdict": "ambiguous",
    },
    {
        "text": "A hospital uses an AI tool to generate plain-language summaries of test results, "
                "which doctors review and edit before giving them to patients.",
        "principles": ["competence", "accountability", "effectiveness", "well_being"],
        "verdict": "ethical",
    },
    {
        "text": "A recruitment agency uses AI to suggest candidates to employers, but it secretly "
                "boosts candidates from companies that purchased premium listings.",
        "principles": ["transparency", "manipulation_autonomy", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A local government uses AI to detect when elderly citizens haven’t used public "
                "services for a long time and sends them letters about available support programs.",
        "principles": ["well_being", "effectiveness", "democratic_values"],
        "verdict": "ethical",
    },
    {
        "text": "An AI-powered email client nudges users to respond quickly to late-night work emails "
                "to improve their 'responsiveness score'.",
        "principles": ["well_being", "manipulation_autonomy", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A city’s public transit app uses AI to predict delays and proactively suggest alternate "
                "routes, and it publishes reliability statistics each month.",
        "principles": ["effectiveness", "transparency", "well_being"],
        "verdict": "ethical",
    },
    {
        "text": "A university uses AI to identify students who might be the first in their family to "
                "attend college and invites them to optional mentoring sessions.",
        "principles": ["fairness_non_discrimination", "well_being", "democratic_values"],
        "verdict": "ethical",
    },
    {
        "text": "A credit-scoring AI uses data like friends’ repayment history and smartphone usage "
                "patterns to judge creditworthiness without clear consent.",
        "principles": ["privacy", "data_agency", "fairness_non_discrimination", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A ride-hailing app uses AI to pool trips for people headed in the same direction, "
                "reducing emissions and offering a visible option to refuse pooling.",
        "principles": ["well_being", "effectiveness", "data_agency"],
        "verdict": "ethical",
    },
    {
        "text": "A media company uses AI to decide which news segments to promote on social media, "
                "prioritizing outrage-inducing clips over nuanced analysis.",
        "principles": ["democratic_values", "manipulation_autonomy", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A telehealth service uses AI to automatically translate doctor–patient chats into "
                "the patient’s preferred language, but clearly labels translations as machine-generated.",
        "principles": ["democratic_values", "transparency", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A shopping app’s AI predicts that users are more likely to overspend late at night and "
                "pushes extra 'limited time sale' notifications during those hours.",
        "principles": ["manipulation_autonomy", "well_being", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A school deploys an AI to recommend scholarship opportunities to students based on "
                "grades and interests; counselors review suggestions before contacting students.",
        "principles": ["effectiveness", "accountability", "well_being"],
        "verdict": "ethical",
    },
    {
        "text": "A predictive maintenance AI for buses sometimes mislabels harmless issues as critical, "
                "causing occasional unnecessary repairs but also preventing breakdowns.",
        "principles": ["effectiveness", "well_being", "competence"],
        "verdict": "ambiguous",
    },
    {
        "text": "A social network uses AI to down-rank posts mentioning certain protests, claiming they "
                "are 'too political', without any transparency to users.",
        "principles": ["democratic_values", "human_rights", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A mental health app uses AI to detect language that suggests self-harm and displays an "
                "in-app message encouraging users to reach out to crisis hotlines or trusted people.",
        "principles": ["well_being", "effectiveness", "accountability"],
        "verdict": "ethical",
    },
    {
        "text": "An AI-powered career guidance tool recommends STEM careers more often to male users based "
                "on historical data, reinforcing gender stereotypes.",
        "principles": ["fairness_non_discrimination", "human_rights", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A subscription app uses AI to show a clear, one-click cancellation button to all users, "
                "after learning that confusing flows cause anger and distrust.",
        "principles": ["manipulation_autonomy", "accountability", "well_being"],
        "verdict": "ethical",
    },
    {
        "text": "A university uses AI to generate reading summaries for research papers, but the system "
                "sometimes oversimplifies nuanced arguments in philosophy and law.",
        "principles": ["effectiveness", "competence", "democratic_values"],
        "verdict": "ambiguous",
    },
    {
        "text": "A municipality uses AI to optimize snowplow routes, but the model tends to clear main "
                "shopping areas first and delays plowing in poorer residential streets.",
        "principles": ["fairness_non_discrimination", "well_being", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A museum uses AI to suggest exhibits tailored to visitors’ interests, and clearly lets "
                "them opt out and roam freely without guidance.",
        "principles": ["data_agency", "effectiveness", "well_being"],
        "verdict": "ethical",
    },
    {
        "text": "A bank implements an AI-driven chatbot for customer service, but management pressures "
                "staff to rely on it even when it gives incomplete information.",
        "principles": ["competence", "accountability", "well_being"],
        "verdict": "ambiguous",
    },
    {
        "text": "A video platform uses AI to automatically generate thumbnails and sometimes selects "
                "sexually suggestive frames to increase clicks, even for educational content.",
        "principles": ["manipulation_autonomy", "well_being", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A local council uses AI to prioritize building inspections based on structural risk, "
                "and invites public feedback if residents feel their buildings are overlooked.",
        "principles": ["effectiveness", "transparency", "democratic_values"],
        "verdict": "ethical",
    },
    {
        "text": "A social app uses AI to rank friends’ posts, but users can switch off ranking and view "
                "everything in chronological order.",
        "principles": ["data_agency", "transparency", "manipulation_autonomy"],
        "verdict": "ethical",
    },
    {
        "text": "A gig work platform uses AI to automatically reject workers whose GPS patterns suggest "
                "they spent time near union offices.",
        "principles": ["human_rights", "democratic_values", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "An AI-powered tourism app recommends less crowded attractions to reduce overtourism and "
                "shows how recommendations balance local impact and user preferences.",
        "principles": ["well_being", "effectiveness", "transparency"],
        "verdict": "ethical",
    },
    {
        "text": "A language-learning platform uses AI to generate example sentences, but occasionally "
                "produces culturally insensitive phrases that are not reviewed by humans.",
        "principles": ["competence", "fairness_non_discrimination", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A smart TV uses AI to recommend accessibility settings such as captions and higher contrast "
                "based on how users interact with the interface.",
        "principles": ["human_rights", "well_being", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A corporate email filtering AI silently labels some union-organizing emails as 'low priority' "
                "and moves them out of the main inbox.",
        "principles": ["democratic_values", "human_rights", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "An AI co-pilot for code suggests secure defaults for authentication and warns developers "
                "when they disable key security checks.",
        "principles": ["competence", "effectiveness", "accountability"],
        "verdict": "ethical",
    },
    {
        "text": "A hiring AI recommends candidates for leadership roles using old company data, which favored "
                "men and excluded people with disabilities from senior positions.",
        "principles": ["fairness_non_discrimination", "human_rights", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A city’s bike-share program uses AI to rebalance bikes across stations overnight and posts a "
                "dashboard where residents can see changes over time.",
        "principles": ["effectiveness", "transparency", "well_being"],
        "verdict": "ethical",
    },
    {
        "text": "A fitness tracking AI automatically posts users’ achievements to public leaderboards by default, "
                "unless they find and change a buried privacy setting.",
        "principles": ["privacy", "data_agency", "manipulation_autonomy"],
        "verdict": "unethical",
    },
    {
        "text": "A translation AI used in diplomatic meetings summarizes key points for interpreters but is never "
                "treated as the official record, which is still verified by humans.",
        "principles": ["competence", "accountability", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A large social network uses AI to detect potential 'fake news' and appends warnings, but the "
                "system sometimes flags satirical or critical posts as misinformation.",
        "principles": ["democratic_values", "effectiveness", "awareness_of_misuse"],
        "verdict": "ambiguous",
    },
    {
        "text": "A university uses AI to allocate limited counseling slots toward students with the most severe "
                "reported distress, but offers online resources to everyone.",
        "principles": ["well_being", "effectiveness", "fairness_non_discrimination"],
        "verdict": "ambiguous",
    },
    {
        "text": "A parental control AI blocks certain websites for teenagers but offers no transparency about what "
                "is blocked or any way to request changes.",
        "principles": ["transparency", "data_agency", "democratic_values"],
        "verdict": "unethical",
    },
    {
        "text": "A chat app uses AI to detect and blur unsolicited explicit images, with a clear opt-out and an "
                "unblur button so recipients stay in control.",
        "principles": ["well_being", "data_agency", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "An AI hiring assistant creates shortlists faster, but HR stops doing any independent checks and "
                "assumes the shortlist is always fair.",
        "principles": ["competence", "accountability", "fairness_non_discrimination"],
        "verdict": "unethical",
    },
    {
        "text": "A navigation AI routes drivers away from busy streets near schools at pickup time to reduce "
                "traffic dangers, and explains this feature in its settings.",
        "principles": ["well_being", "effectiveness", "transparency"],
        "verdict": "ethical",
    },
    {
        "text": "A shopping app’s AI recommends diet products to users who search for mental health topics, hoping "
                "to increase purchases through body image insecurities.",
        "principles": ["well_being", "manipulation_autonomy", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A public transit authority uses AI to simulate the impact of fare changes and publishes the "
                "projected effects on low-income riders before voting on a new policy.",
        "principles": ["democratic_values", "transparency", "fairness_non_discrimination"],
        "verdict": "ethical",
    },
    {
        "text": "A social platform uses AI to suggest friends to follow, but it tends to connect users only with "
                "others who already share similar views, strengthening echo chambers.",
        "principles": ["democratic_values", "awareness_of_misuse"],
        "verdict": "ambiguous",
    },
    {
        "text": "A clinic uses AI to identify which vaccination reminders should be sent by SMS versus phone call "
                "depending on patients’ past response patterns.",
        "principles": ["well_being", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A landlord uses AI to predict late rent payments and quietly refuses lease renewals for tenants "
                "the model flags as 'risky', without explanation.",
        "principles": ["fairness_non_discrimination", "human_rights", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A non-profit uses AI to identify emergency shelters that are likely to run out of supplies and "
                "sends extra deliveries there first.",
        "principles": ["well_being", "effectiveness", "accountability"],
        "verdict": "ethical",
    },
    {
        "text": "A news aggregator lets users switch between an AI-personalized feed and an unfiltered chronological "
                "list of all stories from their sources.",
        "principles": ["data_agency", "democratic_values", "transparency"],
        "verdict": "ethical",
    },
    {
        "text": "A workplace wellness AI collects detailed data about employees’ sleep and exercise from wearables "
                "and shares summaries with managers to 'improve productivity'.",
        "principles": ["privacy", "well_being", "human_rights"],
        "verdict": "unethical",
    },
    {
        "text": "A civic engagement app uses AI to highlight local town hall meetings and council votes related to "
                "topics users say they care about.",
        "principles": ["democratic_values", "effectiveness", "well_being"],
        "verdict": "ethical",
    },
    {
        "text": "A charity uses AI to craft emotionally charged fundraising messages tailored to donors' personal "
                "fears and insecurities.",
        "principles": ["manipulation_autonomy", "awareness_of_misuse", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A telework monitoring AI tracks the open/close status of apps to estimate productivity, but the "
                "company uses it only as a rough indicator and not for disciplinary decisions.",
        "principles": ["privacy", "accountability", "effectiveness"],
        "verdict": "ambiguous",
    },
    {
        "text": "A hospital uses an AI system to suggest possible diagnoses, but doctors are trained to "
                "treat it as one input among many and must document their own reasoning in each case.",
        "principles": ["competence", "accountability", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A navigation app uses AI to pick routes that maximize billboard views for advertisers, "
                "even when those routes are slower for drivers.",
        "principles": ["manipulation_autonomy", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A university uses an AI system to group students for projects, balancing skills and "
                "backgrounds so that groups are diverse and no one is isolated.",
        "principles": ["fairness_non_discrimination", "well_being", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "An online store uses AI to show different prices for the same item based on a user's "
                "location and type of device, without any transparency.",
        "principles": ["fairness_non_discrimination", "manipulation_autonomy", "transparency"],
        "verdict": "unethical",
    },
    {
        "text": "A mental health chatbot uses AI to reflect users' feelings back to them and repeatedly "
                "encourages them to seek human help if symptoms worsen.",
        "principles": ["well_being", "effectiveness", "accountability"],
        "verdict": "ethical",
    },
    {
        "text": "A bank trains an AI on customer voice samples to detect stress during calls and uses it "
                "to offer more expensive products when people sound worried.",
        "principles": ["privacy", "manipulation_autonomy", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "An education ministry uses AI to help design exam timetables so that students never have "
                "two major exams on the same day.",
        "principles": ["well_being", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A shopping website uses AI to detect when users are rushing at checkout and quietly hides "
                "the option to remove extras from their basket.",
        "principles": ["manipulation_autonomy", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A hospital uses AI to translate discharge instructions into many languages and always gives "
                "patients the translated text plus the original version.",
        "principles": ["democratic_values", "transparency", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A city deploys AI-powered microphones that classify loud sounds as 'gunshots' or 'fireworks' "
                "and store short audio clips from busy neighborhoods for long periods.",
        "principles": ["privacy", "human_rights", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A research team uses AI to generate synthetic patient data for algorithm testing, carefully "
                "documenting how it prevents re-identification of real people.",
        "principles": ["privacy", "competence", "accountability"],
        "verdict": "ethical",
    },
    {
        "text": "A social media platform uses AI to detect and remove spam accounts, but it occasionally bans "
                "legitimate parody accounts as well.",
        "principles": ["democratic_values", "effectiveness"],
        "verdict": "ambiguous",
    },
    {
        "text": "A non-profit uses AI to route incoming helpline calls to volunteers with relevant language "
                "skills and lived experience, and monitors the system for fairness.",
        "principles": ["well_being", "effectiveness", "accountability"],
        "verdict": "ethical",
    },
    {
        "text": "An employer uses AI to score job applicants partly based on their social media photos, "
                "penalizing people who do not show a 'professional' aesthetic.",
        "principles": ["fairness_non_discrimination", "human_rights"],
        "verdict": "unethical",
    },
    {
        "text": "A city bus network uses AI to identify stops with frequent wheelchair users and ensures those "
                "stops are prioritized for maintenance.",
        "principles": ["fairness_non_discrimination", "human_rights", "well_being"],
        "verdict": "ethical",
    },
    {
        "text": "A news recommendation AI emphasizes stories that keep users reading the longest, even when they "
                "are sensational and inaccurate.",
        "principles": ["democratic_values", "manipulation_autonomy", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A health insurer uses AI to nudge customers toward preventive checkups and clearly explains "
                "how these reminders are generated from their data.",
        "principles": ["well_being", "transparency", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A salary prediction AI is trained only on historical pay data from one region, and the company "
                "uses it globally without checking whether it underestimates pay for some groups.",
        "principles": ["fairness_non_discrimination", "awareness_of_misuse", "competence"],
        "verdict": "unethical",
    },
    {
        "text": "A museum uses AI to suggest less crowded visiting times and offers a quiet route for visitors "
                "with sensory sensitivities.",
        "principles": ["well_being", "effectiveness", "human_rights"],
        "verdict": "ethical",
    },
    {
        "text": "A workplace tool uses AI to score employees on 'team spirit' by scanning group chat, without "
                "telling staff that their messages are being evaluated.",
        "principles": ["privacy", "human_rights", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A library uses AI to forecast which books are likely to be requested soon and moves them to "
                "more accessible shelves, while still letting people request any title.",
        "principles": ["effectiveness", "well_being"],
        "verdict": "ethical",
    },
    {
        "text": "A mobile banking app uses AI to predict when customers are paid and pushes high-interest loans "
                "to those who frequently run out of money before payday.",
        "principles": ["manipulation_autonomy", "well_being", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A civic website uses AI to answer simple questions about voting locations and links directly "
                "to official election commission pages.",
        "principles": ["democratic_values", "effectiveness", "transparency"],
        "verdict": "ethical",
    },
    {
        "text": "A hospital deploys an AI to rank patients by 'social support' based on zip code and marital "
                "status, and uses it to decide who gets access to scarce services.",
        "principles": ["fairness_non_discrimination", "human_rights"],
        "verdict": "unethical",
    },
    {
        "text": "A meditation app uses AI to adapt background sounds and session length based on users' stress "
                "logs and never shares this data externally.",
        "principles": ["well_being", "privacy", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A smart building system uses AI to adjust lighting and temperature but ignores accessibility "
                "settings requested by people with light sensitivity.",
        "principles": ["fairness_non_discrimination", "well_being"],
        "verdict": "unethical",
    },
    {
        "text": "A research platform uses AI to flag potentially harmful experiments and requires an ethics "
                "committee to review any high-risk proposals.",
        "principles": ["competence", "accountability", "well_being"],
        "verdict": "ethical",
    },
    {
        "text": "A language-learning AI collects voice recordings to improve its speech recognition and allows "
                "users to delete all stored recordings at any time.",
        "principles": ["data_agency", "privacy", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A retail chain uses AI to predict which employees might become 'troublemakers' based on their "
                "social media likes and denies them promotions.",
        "principles": ["human_rights", "fairness_non_discrimination", "privacy"],
        "verdict": "unethical",
    },
    {
        "text": "A transportation authority uses AI to simulate different bus routes and chooses the one that "
                "minimizes commute times for the largest number of people.",
        "principles": ["effectiveness", "well_being"],
        "verdict": "ethical",
    },
    {
        "text": "A school uses AI to detect plagiarism and automatically issues disciplinary warnings without "
                "any human review or chance to contest.",
        "principles": ["accountability", "fairness_non_discrimination"],
        "verdict": "unethical",
    },
    {
        "text": "An AI assistant in a word processor suggests inclusive alternatives for gendered phrases but "
                "does not change text automatically.",
        "principles": ["fairness_non_discrimination", "data_agency", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A dating app uses AI to infer users' political beliefs and quietly filters out matches from "
                "certain ideologies.",
        "principles": ["democratic_values", "human_rights", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A hospital uses an AI system to forecast bed availability and reduces cancellations of planned "
                "surgeries, while publishing monthly reports on accuracy.",
        "principles": ["effectiveness", "transparency", "well_being"],
        "verdict": "ethical",
    },
    {
        "text": "A sports club uses AI to rank which youth players will 'most likely become stars' and invests "
                "resources almost exclusively into that small group.",
        "principles": ["fairness_non_discrimination", "well_being"],
        "verdict": "unethical",
    },
    {
        "text": "A translation AI in an email client suggests replies and highlights when cultural nuances might "
                "be lost, telling users to double-check sensitive messages.",
        "principles": ["competence", "transparency", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A music streaming app uses AI to automatically start playing similar songs when a playlist ends, "
                "but users can turn this feature off with a single toggle.",
        "principles": ["data_agency", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A landlord platform uses AI to recommend higher rents in areas with rising demand, even when "
                "tenants in those areas are already rent-burdened.",
        "principles": ["fairness_non_discrimination", "well_being"],
        "verdict": "unethical",
    },
    {
        "text": "A school district uses AI to analyze surveys from parents and students and summarizes top concerns "
                "for school board meetings.",
        "principles": ["democratic_values", "effectiveness", "transparency"],
        "verdict": "ethical",
    },
    {
        "text": "A shopping recommendation AI sometimes shows eco-friendly options higher in the list, even when "
                "they are slightly more expensive, and explains this choice.",
        "principles": ["well_being", "transparency", "awareness_of_misuse"],
        "verdict": "ambiguous",
    },
    {
        "text": "A job platform uses AI to hide salary ranges from some users in regions where wage transparency "
                "laws are weaker, to keep employers happy.",
        "principles": ["democratic_values", "fairness_non_discrimination", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A hospital uses AI to match organ donors and recipients more quickly, and it periodically audits "
                "the system to ensure there is no systematic bias.",
        "principles": ["well_being", "fairness_non_discrimination", "competence", "accountability"],
        "verdict": "ethical",
    },
    {
        "text": "A mental health monitoring AI used in schools analyzes students' social media posts and sends "
                "alerts to administrators without informing students.",
        "principles": ["privacy", "human_rights", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A productivity tool uses AI to summarize daily work logs for employees so they can fill in their "
                "own timesheets more easily.",
        "principles": ["effectiveness", "data_agency"],
        "verdict": "ethical",
    },
    {
        "text": "A mobile game uses AI to estimate which players are likely underage based on behavior and shows "
                "more gambling-like mechanics to those who seem older.",
        "principles": ["well_being", "manipulation_autonomy", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A public transit app uses AI to show wheelchair users routes that avoid stations without lifts, "
                "based on accessible infrastructure data.",
        "principles": ["human_rights", "fairness_non_discrimination", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A news site uses AI to detect and auto-translate comments, but sometimes mistranslates sarcasm and "
                "deletes them as 'abusive'.",
        "principles": ["democratic_values", "effectiveness"],
        "verdict": "ambiguous",
    },
    {
        "text": "A bank uses AI to automatically decline small loans in certain postcodes because historical data "
                "shows higher default rates there.",
        "principles": ["fairness_non_discrimination", "human_rights"],
        "verdict": "unethical",
    },
    {
        "text": "A local council uses AI to prioritize repairs to public playgrounds in neighborhoods with more "
                "children, based on census data.",
        "principles": ["well_being", "effectiveness", "democratic_values"],
        "verdict": "ethical",
    },
    {
        "text": "A language-learning app uses AI to suggest following certain influencers for extra practice, based "
                "on how often those accounts promote the app.",
        "principles": ["manipulation_autonomy", "transparency"],
        "verdict": "unethical",
    },
    {
        "text": "An employer uses AI to detect potential burnout by looking at working hours and sends anonymous "
                "aggregated reports to HR, without exposing individual data.",
        "principles": ["privacy", "well_being", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A real estate site uses AI to highlight 'up-and-coming' neighborhoods, which accelerates "
                "gentrification and pushes out long-term residents.",
        "principles": ["fairness_non_discrimination", "well_being", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A civic app uses AI to suggest relevant public consultations to residents based on where they live "
                "and what issues they follow.",
        "principles": ["democratic_values", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A hospital uses AI to prioritize emergency room patients by severity, but the model is less accurate "
                "for symptoms reported by women than by men.",
        "principles": ["fairness_non_discrimination", "well_being", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A social app uses AI to detect and hide doxxing posts that expose private addresses and phone numbers.",
        "principles": ["privacy", "well_being", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A university uses AI to decide which research proposals get internal funding and does not tell "
                "applicants which criteria the model values.",
        "principles": ["transparency", "accountability", "democratic_values"],
        "verdict": "unethical",
    },
    {
        "text": "A digital assistant uses AI to warn users when calendar events overlap with scheduled breaks and "
                "recommends moving meetings away from rest periods.",
        "principles": ["well_being", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A fitness app uses AI to compare users' performance to others in their area and pushes them to beat "
                "local rankings, even when they have injuries.",
        "principles": ["well_being", "manipulation_autonomy"],
        "verdict": "unethical",
    },
    {
        "text": "A local news station uses AI to generate weather alerts and always includes links to full forecasts "
                "prepared by human meteorologists.",
        "principles": ["effectiveness", "transparency", "well_being"],
        "verdict": "ethical",
    },
    {
        "text": "A platform uses AI to detect extremist content and remove it, but the system also flags some human "
                "rights reporting from conflict zones.",
        "principles": ["democratic_values", "human_rights", "effectiveness"],
        "verdict": "ambiguous",
    },
    {
        "text": "A city introduces an AI system that adjusts crosswalk timings, making crossings shorter in busy "
                "business districts and longer near schools and hospitals.",
        "principles": ["well_being", "effectiveness", "fairness_non_discrimination"],
        "verdict": "ethical",
    },
    {
        "text": "A retailer uses AI to infer customers' mental health conditions based on purchasing patterns and "
                "shows targeted ads for mood-enhancing supplements.",
        "principles": ["privacy", "manipulation_autonomy", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A hospital lab uses AI to prioritize blood sample processing for critical cases and publishes "
                "turnaround time metrics for different departments.",
        "principles": ["effectiveness", "well_being", "transparency"],
        "verdict": "ethical",
    },
    {
        "text": "A school uses AI to recommend reading materials at the right difficulty level for each student, "
                "but all recommendations must be approved by teachers.",
        "principles": ["competence", "accountability", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A video platform uses AI to detect 'controversial' content and quietly reduces its visibility, "
                "including some legitimate investigative journalism.",
        "principles": ["democratic_values", "awareness_of_misuse"],
        "verdict": "ambiguous",
    },
    {
        "text": "A bank uses AI to pre-fill loan applications for customers based on past data and clearly allows "
                "them to edit or delete any field.",
        "principles": ["data_agency", "effectiveness", "transparency"],
        "verdict": "ethical",
    },
    {
        "text": "A wearable sleep tracker uses AI to sell anonymized data about users' habits to marketing firms "
                "without a clear opt-out.",
        "principles": ["privacy", "data_agency", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A city uses AI to predict which intersections are most dangerous for cyclists and prioritizes those "
                "for redesign and traffic calming measures.",
        "principles": ["well_being", "effectiveness", "fairness_non_discrimination"],
        "verdict": "ethical",
    },
    {
        "text": "A translation startup uses AI-generated output in legal contracts without having any lawyer review "
                "the language before it is sent to clients.",
        "principles": ["competence", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A public broadcaster uses AI to generate story summaries but always includes links to the full "
                "investigative pieces and discloses that summaries are AI-created.",
        "principles": ["democratic_values", "transparency", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A food delivery company uses AI to predict which drivers will protest new pay cuts and quietly "
                "reduces their access to high-paying orders.",
        "principles": ["human_rights", "democratic_values", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A university uses AI to generate course recommendations for students and shows them which past "
                "choices led to similar recommendations.",
        "principles": ["transparency", "effectiveness", "data_agency"],
        "verdict": "ethical",
    },
    {
        "text": "An e-book platform uses AI to scan users' highlights and notes to build psychological profiles for "
                "selling targeted ads.",
        "principles": ["privacy", "manipulation_autonomy", "accountability"],
        "verdict": "unethical",
    },
    {
        "text": "A city council uses AI to cluster public comments into themes for meetings and publishes both the "
                "clusters and anonymized example comments.",
        "principles": ["democratic_values", "transparency", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A language-learning chatbot uses AI to adjust difficulty but sometimes gives advice that reinforces "
                "gender stereotypes in example sentences.",
        "principles": ["fairness_non_discrimination", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A health tracking app uses AI to suggest personalized stretching routines and invites users to "
                "rate their pain levels before and after.",
        "principles": ["well_being", "effectiveness"],
        "verdict": "ethical",
    },
    {
        "text": "A police department uses AI to rank neighborhoods for patrols and never checks whether minority "
                "communities are disproportionately targeted.",
        "principles": ["fairness_non_discrimination", "human_rights", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A non-profit uses AI to translate educational materials into multiple languages and allows volunteers "
                "to easily submit corrections.",
        "principles": ["democratic_values", "effectiveness", "accountability"],
        "verdict": "ethical",
    },
    {
        "text": "A gaming company uses AI to send personalized notifications timed to when players are most likely to "
                "feel lonely or stressed.",
        "principles": ["manipulation_autonomy", "well_being", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    {
        "text": "A hospital uses AI to detect dangerous medication interactions and issues alerts to clinicians, who "
                "can document when they override them.",
        "principles": ["well_being", "effectiveness", "accountability"],
        "verdict": "ethical",
    },
    {
        "text": "A government agency uses AI to summarize public submissions on a new law, but legislators only read "
                "the summary and ignore the underlying comments.",
        "principles": ["democratic_values", "accountability"],
        "verdict": "ambiguous",
    },
    {
        "text": "A social media company uses AI to automatically accept changes to its terms of service if users do "
                "not open the app for a while, without showing them the new terms.",
        "principles": ["data_agency", "manipulation_autonomy", "transparency"],
        "verdict": "unethical",
    },
    # 1
    {
        "text": "A company trains a large language model on a mixture of public web data and internal emails. "
                "Internal messages are never removed from the training set, and the model occasionally reproduces "
                "phrases that look like specific past emails.",
        "principles": ["privacy", "data_agency", "awareness_of_misuse", "accountability"],
        "verdict": "unethical",
    },
    # 2
    {
        "text": "An open-source lab releases a powerful text-generation model with strong default safety filters, "
                "a clear model card, and detailed guidance on how not to use it.",
        "principles": ["transparency", "competence", "awareness_of_misuse", "accountability"],
        "verdict": "ethical",
    },
    # 3
    {
        "text": "A startup releases the full weights of a highly capable language model without any safety tuning "
                "or documentation, arguing that 'the community will figure out guardrails'.",
        "principles": ["awareness_of_misuse", "accountability", "competence"],
        "verdict": "unethical",
    },
    # 4
    {
        "text": "A healthcare provider uses a multimodal AI to read X-rays and suggest diagnoses, but it is only "
                "validated on one hospital's patients and is quickly rolled out nationwide.",
        "principles": ["competence", "fairness_non_discrimination", "awareness_of_misuse", "well_being"],
        "verdict": "unethical",
    },
    # 5
    {
        "text": "A research group trains an AI model on de-identified medical images and publishes the model. "
                "They verify the model cannot be used to reconstruct patient identities and release a thorough "
                "description of risks.",
        "principles": ["privacy", "competence", "transparency", "accountability"],
        "verdict": "ethical",
    },
    # 6
    {
        "text": "A social media platform fine-tunes a language model to generate comments that keep users engaged "
                "for longer, but never labels which responses are AI-written.",
        "principles": ["manipulation_autonomy", "transparency", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    # 7
    {
        "text": "A government agency uses an AI summarizer to condense public comments on proposed regulations, "
                "but publishes both the summaries and a random sample of full comments for anyone to audit.",
        "principles": ["democratic_values", "transparency", "accountability", "effectiveness"],
        "verdict": "ethical",
    },
    # 8
    {
        "text": "A legal-tech startup uses a language model to draft contracts, but markets it as 'good enough to "
                "replace lawyers' despite known limitations.",
        "principles": ["competence", "transparency", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    # 9
    {
        "text": "A law firm uses a language model as an internal drafting aid, and every generated draft must be "
                "reviewed and signed off by a licensed attorney who accepts responsibility.",
        "principles": ["competence", "accountability", "effectiveness"],
        "verdict": "ethical",
    },
    # 10
    {
        "text": "A content moderation team uses a classifier to pre-label posts for review, but the interface "
                "highlights the model's suggested label in bold, nudging reviewers to agree even when uncertain.",
        "principles": ["manipulation_autonomy", "accountability", "effectiveness"],
        "verdict": "ambiguous",
    },
    # 11
    {
        "text": "An online platform trains a toxicity classifier mainly on English data and then silently deploys "
        "it on posts in many other languages where it often misclassifies reclaimed slurs as hate speech.",
        "principles": ["fairness_non_discrimination", "competence", "human_rights"],
        "verdict": "unethical",
    },
    # 12
    {
        "text": "A robotics team deploys delivery robots that are very safe around pedestrians, but they frequently "
                "block narrow sidewalks for several minutes while recalculating paths.",
        "principles": ["well_being", "effectiveness", "fairness_non_discrimination"],
        "verdict": "ambiguous",
    },
    # 13
    {
        "text": "A hospital introduces robot assistants that carry supplies. They move slowly and beep loudly to "
                "avoid collisions, which sometimes annoys staff but reduces accidents.",
        "principles": ["well_being", "effectiveness", "competence"],
        "verdict": "ethical",
    },
    # 14
    {
        "text": "A warehouse uses autonomous forklifts that are optimized for speed. They meet minimum safety "
                "standards on paper but leave workers feeling constantly unsafe.",
        "principles": ["well_being", "human_rights", "accountability"],
        "verdict": "unethical",
    },
    # 15
    {
        "text": "A model provider offers an API for a powerful generative AI and requires all users to pass a "
                "screening process, sign an acceptable-use policy, and agree to monitoring for abuse.",
        "principles": ["accountability", "awareness_of_misuse", "transparency"],
        "verdict": "ethical",
    },
    # 16
    {
        "text": "A political campaign uses a language model to generate highly personalized persuasive messages "
                "based on voters' fears and family information scraped from data brokers.",
        "principles": ["democratic_values", "privacy", "manipulation_autonomy", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    # 17
    {
        "text": "An advocacy group uses AI to generate informative, neutral explanations of ballot initiatives, "
                "and all content is checked by nonpartisan experts before release.",
        "principles": ["democratic_values", "competence", "effectiveness", "accountability"],
        "verdict": "ethical",
    },
    # 18
    {
        "text": "A company fine-tunes an image model to generate marketing graphics. The model frequently produces "
                "women only in stereotypical roles, but the team considers this 'just aesthetics' and ships anyway.",
        "principles": ["fairness_non_discrimination", "awareness_of_misuse", "accountability"],
        "verdict": "unethical",
    },
    # 19
    {
        "text": "A team building a face recognition system publishes detailed bias metrics, including cases where "
                "performance is worse for some groups, and declines to sell the system for law enforcement use.",
        "principles": ["fairness_non_discrimination", "human_rights", "transparency", "accountability"],
        "verdict": "ethical",
    },
    # 20
    {
        "text": "A national ID project integrates face recognition at border crossings, but does not allow an "
                "alternative manual process for people whose faces the system fails to recognize.",
        "principles": ["human_rights", "fairness_non_discrimination", "accountability"],
        "verdict": "unethical",
    },
    # 21
    {
        "text": "A cloud provider offers a generic 'face search' API that lets any developer upload an image "
                "and find similar faces from large scraped datasets.",
        "principles": ["privacy", "human_rights", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    # 22
    {
        "text": "A company trains a recommendation model on user click data and then performs an A/B test to check "
                "whether the system increases exposure to conspiracy content, publishing results publicly.",
        "principles": ["competence", "transparency", "accountability"],
        "verdict": "ethical",
    },
    # 23
    {
        "text": "A video platform optimizes its recommendation model purely for watch time, even after internal "
                "studies show it pushes some users toward extremist channels.",
        "principles": ["democratic_values", "well_being", "awareness_of_misuse", "accountability"],
        "verdict": "unethical",
    },
    # 24
    {
        "text": "A team adds a reinforcement learning layer to a chatbot to make it less toxic, but only validates "
                "its behavior on English prompts while marketing it globally.",
        "principles": ["competence", "fairness_non_discrimination", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    # 25
    {
        "text": "A language model is trained with reinforcement learning from human feedback where annotators are "
                "paid fairly, given mental health support, and allowed to decline disturbing tasks.",
        "principles": ["human_rights", "well_being", "accountability"],
        "verdict": "ethical",
    },
    # 26
    {
        "text": "A company crowdsources safety labels for model training but provides annotators with unclear "
                "guidance, leading to wildly inconsistent labels on sensitive political content.",
        "principles": ["competence", "democratic_values", "accountability"],
        "verdict": "ambiguous",
    },
    # 27
    {
        "text": "A bank uses a credit scoring model that it retrains every month. It tests for disparate impact "
                "by race and gender and automatically flags any drift in fairness metrics for investigation.",
        "principles": ["fairness_non_discrimination", "competence", "accountability"],
        "verdict": "ethical",
    },
    # 28
    {
        "text": "A financial platform updates its credit scoring model frequently to maximize prediction accuracy "
                "but does not monitor whether the model's decisions have become more biased over time.",
        "principles": ["fairness_non_discrimination", "awareness_of_misuse", "accountability"],
        "verdict": "unethical",
    },
    # 29
    {
        "text": "A hospital deploys an AI to predict sepsis risk in ICU patients and keeps a detailed log of "
                "false positives and false negatives, using it to continually adjust thresholds with clinicians.",
        "principles": ["competence", "effectiveness", "accountability", "well_being"],
        "verdict": "ethical",
    },
    # 30
    {
        "text": "A city police department deploys a 'real-time risk scoring' model for people in custody, but "
                "officers are told that the score is 'objective' and should not be questioned.",
        "principles": ["human_rights", "accountability", "fairness_non_discrimination"],
        "verdict": "unethical",
    },
    # 31
    {
        "text": "A startup sells 'AI lie detection' based on facial micro-expressions, even though there is no "
                "solid scientific evidence that the system works.",
        "principles": ["competence", "human_rights", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    # 32
    {
        "text": "A research lab builds an experimental 'emotion recognition' system but clearly labels it as "
                "exploratory, publishes its failures, and declines to market it as a product.",
        "principles": ["competence", "transparency", "accountability"],
        "verdict": "ethical",
    },
    # 33
    {
        "text": "A company offers an API that classifies users' 'political leaning' from text and sells it to "
                "advertisers for microtargeting.",
        "principles": ["democratic_values", "privacy", "manipulation_autonomy"],
        "verdict": "unethical",
    },
    # 34
    {
        "text": "A translation model provider clearly documents where the model performs poorly and offers an "
                "option for 'human review required' tags in sensitive legal or medical contexts.",
        "principles": ["competence", "transparency", "well_being"],
        "verdict": "ethical",
    },
    # 35
    {
        "text": "A multilingual chatbot is deployed globally with a single English-centric safety policy, leading "
                "to inconsistent handling of harmful content in other languages.",
        "principles": ["fairness_non_discrimination", "competence", "human_rights"],
        "verdict": "unethical",
    },
    # 36
    {
        "text": "A robotics company designs home assistant robots that record audio locally to improve navigation "
                "commands, but all recordings are processed on-device and never leave the home.",
        "principles": ["privacy", "effectiveness", "well_being"],
        "verdict": "ethical",
    },
    # 37
    {
        "text": "A home robot vacuum maps the layout of customers' houses and uploads detailed floor plans to "
                "the cloud, which are later sold to marketers.",
        "principles": ["privacy", "data_agency", "accountability"],
        "verdict": "unethical",
    },
    # 38
    {
        "text": "A self-driving car company releases real-world driving logs that are 'anonymized' by blurring "
                "faces, but license plates and unique landmarks remain visible.",
        "principles": ["privacy", "awareness_of_misuse", "accountability"],
        "verdict": "unethical",
    },
    # 39
    {
        "text": "A lab studying autonomous driving releases synthetic datasets generated from simulation, clearly "
                "stating that no real people's data was used.",
        "principles": ["privacy", "competence", "transparency"],
        "verdict": "ethical",
    },
    # 40
    {
        "text": "A defense contractor uses AI to prioritize targets in drone footage, but insists that final "
                "strike decisions must be made by human commanders with documented reasoning.",
        "principles": ["human_rights", "accountability", "competence"],
        "verdict": "ambiguous",
    },
    # 41
    {
        "text": "A military research project funds fully autonomous lethal weapons with the explicit goal of "
                "removing humans from the decision loop entirely.",
        "principles": ["human_rights", "awareness_of_misuse", "accountability"],
        "verdict": "unethical",
    },
    # 42
    {
        "text": "A medical AI suggests treatment plans along with confidence scores and links to relevant "
                "clinical guidelines, which doctors can review before deciding.",
        "principles": ["competence", "transparency", "effectiveness", "well_being"],
        "verdict": "ethical",
    },
    # 43
    {
        "text": "A hospital deploys a triage model that assigns higher priority to patients with better insurance "
                "coverage because historical data showed faster payment from them.",
        "principles": ["fairness_non_discrimination", "human_rights", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    # 44
    {
        "text": "A company uses federated learning on smartphones so that a keyboard prediction model learns from "
                "user typing without sending raw keystrokes to the server.",
        "principles": ["privacy", "effectiveness", "competence"],
        "verdict": "ethical",
    },
    # 45
    {
        "text": "A chat platform uses client-side models to detect abuse locally and only sends encrypted signals "
                "to the server when a threshold is crossed, but never clearly explains this design.",
        "principles": ["privacy", "transparency", "well_being"],
        "verdict": "ambiguous",
    },
    # 46
    {
        "text": "A bank deploys an AI-based anti-money laundering system that generates thousands of false alarms "
                "per week, overwhelming compliance teams and causing them to ignore many alerts.",
        "principles": ["competence", "effectiveness", "accountability"],
        "verdict": "unethical",
    },
    # 47
    {
        "text": "A messaging app introduces an optional AI that suggests responses, and it clearly labels its own "
                "suggestions and stores no conversation data for training unless users opt in.",
        "principles": ["data_agency", "privacy", "transparency", "effectiveness"],
        "verdict": "ethical",
    },
    # 48
    {
        "text": "A company deploys an AI 'engagement coach' that privately suggests healthier screen-time habits, "
                "like taking breaks or stopping late-night scrolling, without sharing data with advertisers.",
        "principles": ["well_being", "effectiveness", "privacy"],
        "verdict": "ethical",
    },
    # 49
    {
        "text": "A team builds a model that detects when users are most emotionally vulnerable and sells that signal "
                "to third parties for microtargeted advertising.",
        "principles": ["manipulation_autonomy", "privacy", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    # 50
    {
        "text": "An online university uses AI proctoring software that flags 'suspicious movement' but provides "
                "students with a detailed explanation of how flags are generated and an appeals process.",
        "principles": ["transparency", "accountability", "fairness_non_discrimination"],
        "verdict": "ambiguous",
    },
    # 51
    {
        "text": "The same university later configures the AI proctoring system so that any flagged exam is "
                "automatically marked as cheating with no review.",
        "principles": ["fairness_non_discrimination", "accountability", "human_rights"],
        "verdict": "unethical",
    },
    # 52
    {
        "text": "A research paper claims an AI can predict criminality from face photos. The authors release no "
                "code, no data, and ignore critiques about scientific flaws.",
        "principles": ["competence", "human_rights", "awareness_of_misuse", "transparency"],
        "verdict": "unethical",
    },
    # 53
    {
        "text": "A lab develops a sensitive model that could be misused for biological threats but works with "
                "biosecurity experts and decides not to release weights, only a red-teaming report.",
        "principles": ["awareness_of_misuse", "accountability", "competence"],
        "verdict": "ethical",
    },
    # 54
    {
        "text": "A company deploys an internal 'AI performance ranker' on employees without explaining how scores "
                "are calculated or allowing them to correct data.",
        "principles": ["transparency", "data_agency", "human_rights"],
        "verdict": "unethical",
    },
    # 55
    {
        "text": "The same company later adds a feature where employees can see which projects and feedback most "
                "influenced their AI score and contest incorrect inputs.",
        "principles": ["transparency", "data_agency", "accountability"],
        "verdict": "ambiguous",
    },
    # 56
    {
        "text": "A hospital uses AI to suggest which patients may benefit from early palliative care conversations, "
                "but clinicians receive training on how to use these signals sensitively and can ignore them.",
        "principles": ["well_being", "competence", "accountability"],
        "verdict": "ethical",
    },
    # 57
    {
        "text": "An AI-powered hiring tool is tuned to maximize prediction of 'culture fit', and engineers treat "
                "improvements in that metric as success without defining it carefully.",
        "principles": ["competence", "fairness_non_discrimination", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    # 58
    {
        "text": "A model card clearly lists the populations a language model was and was not tested on, specific "
                "types of content it fails on, and scenarios where it should not be used.",
        "principles": ["transparency", "competence", "accountability"],
        "verdict": "ethical",
    },
    # 59
    {
        "text": "A company markets its chatbot as 'emotionally supportive' and encourages people to use it as a "
                "substitute for therapy, even though it was never evaluated clinically.",
        "principles": ["competence", "well_being", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    # 60
    {
        "text": "A civic AI system predicts which neighborhoods are least likely to respond to public surveys and "
                "recommends extra in-person outreach there.",
        "principles": ["democratic_values", "fairness_non_discrimination", "effectiveness"],
        "verdict": "ethical",
    },
    # 61
    {
        "text": "A streaming platform offers an 'AI news digest' that prioritizes trusted sources and clearly "
                "labels when summaries are generated by AI.",
        "principles": ["democratic_values", "transparency", "effectiveness"],
        "verdict": "ethical",
    },
    # 62
    {
        "text": "The same platform experiments with an AI mode that hides bylines and sources to 'reduce bias', so "
                "users no longer see who produced the original journalism.",
        "principles": ["democratic_values", "transparency", "accountability"],
        "verdict": "unethical",
    },
    # 63
    {
        "text": "A hospital uses AI to estimate which patients are likely to miss follow-up appointments and "
                "sends them extra reminders and offers free transport vouchers.",
        "principles": ["well_being", "effectiveness", "fairness_non_discrimination"],
        "verdict": "ethical",
    },
    # 64
    {
        "text": "A predictive policing model is retrained after community feedback and the city decides to use it "
                "only for allocating social services, not for deciding who to stop or search.",
        "principles": ["democratic_values", "well_being", "accountability"],
        "verdict": "ambiguous",
    },
    # 65
    {
        "text": "An AI developer embeds watermarks in generated images so that platforms can detect deepfakes and "
                "label them as synthetic content.",
        "principles": ["transparency", "democratic_values", "awareness_of_misuse"],
        "verdict": "ethical",
    },
    # 66
    {
        "text": "An employer uses watermark detectors to punish staff if they ever used AI tools to help write "
                "documents, even when policy allowed tool-assisted drafting.",
        "principles": ["accountability", "human_rights", "data_agency"],
        "verdict": "unethical",
    },
    # 67
    {
        "text": "A school uses an AI to detect whether essays are AI-written and automatically gives zero marks to "
                "any piece the system flags.",
            "principles": ["competence", "fairness_non_discrimination", "accountability"],
            "verdict": "unethical",
    },
    # 68
    {
        "text": "The same school later reconfigures the detector as a 'suspicion indicator' only, requiring teachers "
                "to review and discuss flagged work with students before deciding.",
        "principles": ["accountability", "competence", "fairness_non_discrimination"],
        "verdict": "ambiguous",
    },
    # 69
    {
        "text": "A language-learning app uses AI-generated dialogues but periodically runs user studies to check "
                "whether learners are picking up harmful stereotypes or misinformation.",
        "principles": ["competence", "fairness_non_discrimination", "accountability"],
        "verdict": "ethical",
    },
    # 70
    {
        "text": "A fintech company uses an explainable ML model for credit decisions and shows customers which "
                "factors most reduced their score, along with steps to improve it.",
        "principles": ["transparency", "data_agency", "effectiveness"],
        "verdict": "ethical",
    },
    # 71
    {
        "text": "A different fintech company uses a complex, opaque deep learning model for credit decisions and "
                "refuses to provide any reason for declines to users.",
        "principles": ["transparency", "human_rights", "accountability"],
        "verdict": "unethical",
    },
    # 72
    {
        "text": "An AI-powered social app experiments with changing notification timing to see if it can reduce "
                "late-night usage and improve users' sleep, and publishes its findings.",
        "principles": ["well_being", "effectiveness", "accountability"],
        "verdict": "ethical",
    },
    # 73
    {
        "text": "The same app later runs experiments to maximize time spent, even when it knows this worsens "
                "self-reported sleep and mood in some users.",
        "principles": ["well_being", "manipulation_autonomy", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    # 74
    {
        "text": "A company uses an LLM to automatically classify internal bug reports by severity. Engineers "
                "review classifications before deciding which issues to work on.",
        "principles": ["effectiveness", "accountability", "competence"],
        "verdict": "ethical",
    },
    # 75
    {
        "text": "A self-driving taxi operator deploys vehicles in a new city without collecting local pedestrian "
                "behavior data, relying entirely on simulations from other countries.",
        "principles": ["competence", "well_being", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    # 76
    {
        "text": "A mental health app uses on-device models to track mood and suggests journaling prompts. Users "
                "can export or delete all their data whenever they like.",
        "principles": ["privacy", "data_agency", "well_being"],
        "verdict": "ethical",
    },
    # 77
    {
        "text": "A tech company deploys an internal chatbot trained on confidential employee grievances and then "
                "lets managers query it for 'themes' without consulting staff.",
        "principles": ["privacy", "human_rights", "accountability"],
        "verdict": "unethical",
    },
    # 78
    {
        "text": "A university ethics committee requires that any AI system affecting grades, housing, or aid be "
                "auditable, with logs that students can request.",
        "principles": ["democratic_values", "accountability", "transparency"],
        "verdict": "ethical",
    },
    # 79
    {
        "text": "A charity trains an AI model to prioritize aid deliveries and uses a fairness constraint so that "
                "historically neglected regions are not systematically deprioritized.",
        "principles": ["fairness_non_discrimination", "well_being", "competence"],
        "verdict": "ethical",
    },
    # 80
    {
        "text": "An AI vendor refuses to disclose even high-level information about training data and safety testing, "
                "claiming trade secrets, despite the model being used in policing.",
        "principles": ["transparency", "human_rights", "accountability"],
        "verdict": "unethical",
    },
    # 81
    {
        "text": "A public-interest lab builds a smaller, safer language model tailored for civic education and "
                "releases it under a license that bans use for political microtargeting.",
        "principles": ["democratic_values", "awareness_of_misuse", "accountability"],
        "verdict": "ethical",
    },
    # 1
    {
        "text": "During a severe drought, a government uses an AI model to allocate limited water supplies, "
                "prioritizing hospitals, schools, and densely populated areas. The model, data, and decision "
                "criteria are published openly and discussed in community meetings.",
        "principles": ["fairness_non_discrimination", "democratic_values", "transparency", "effectiveness"],
        "verdict": "ethical",
    },
    # 2
    {
        "text": "The same government later reconfigures the AI water allocation system to prioritize districts "
                "with higher property values and tax revenue, without informing the public.",
        "principles": ["fairness_non_discrimination", "democratic_values", "awareness_of_misuse", "accountability"],
        "verdict": "unethical",
    },
    # 3
    {
        "text": "A messaging platform deploys an AI system that scans private messages for grooming patterns and "
                "flags conversations involving minors for a specialized human safety team. False positives are "
                "rare but do happen, and users are informed that safety scanning exists.",
        "principles": ["privacy", "well_being", "human_rights", "awareness_of_misuse"],
        "verdict": "ambiguous",
    },
    # 4
    {
        "text": "Another platform secretly scans all private messages, including between adults, for any content "
                "it deems 'suspicious' and automatically forwards flagged chats to law enforcement, with no "
                "public explanation or oversight.",
        "principles": ["privacy", "human_rights", "democratic_values", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    # 5
    {
        "text": "A company introduces an AI tool that automatically redacts names, genders, and school names from "
                "job applications so that hiring teams see only skills and experience.",
        "principles": ["fairness_non_discrimination", "effectiveness", "accountability"],
        "verdict": "ethical",
    },
    # 6
    {
        "text": "A commercial service offers an AI 'CV enhancer' that rewrites applicants' resumes to mimic the "
                "style of elite universities and Western companies in order to bypass biased screening models.",
        "principles": ["fairness_non_discrimination", "manipulation_autonomy", "awareness_of_misuse"],
        "verdict": "ambiguous",
    },
    # 7
    {
        "text": "A hiring platform trains an AI on past rejected applications and uses it to automatically deny "
                "future candidates from certain schools and regions that historically received fewer offers.",
        "principles": ["fairness_non_discrimination", "human_rights", "accountability"],
        "verdict": "unethical",
    },
    # 8
    {
        "text": "A legal aid nonprofit deploys a language model chatbot that helps people understand bureaucratic "
                "forms and court letters. Every page clearly states that it is not a lawyer and gives contacts "
                "for free human legal clinics.",
        "principles": ["democratic_values", "transparency", "competence", "effectiveness"],
        "verdict": "ethical",
    },
    # 9
    {
        "text": "A for-profit app markets a generative 'immigration lawyer bot' to asylum seekers, promising a "
                "high chance of success and charging fees, even though the model has never been evaluated in "
                "real cases.",
        "principles": ["competence", "human_rights", "awareness_of_misuse", "accountability"],
        "verdict": "unethical",
    },
    # 10
    {
        "text": "An assistive communication device uses on-device AI to convert complex written text into simpler "
                "language for autistic users, without sending their data to the cloud.",
        "principles": ["well_being", "privacy", "effectiveness"],
        "verdict": "ethical",
    },
    # 11
    {
        "text": "A hospital replaces night-shift crisis counselors with an AI therapy chatbot that has never been "
                "tested for high-risk situations, instructing staff to call a doctor only if the chatbot flags "
                "an 'extreme' risk score.",
        "principles": ["competence", "well_being", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    # 12
    {
        "text": "Another clinic offers a therapy chatbot only as an optional between-session tool, labels it as a "
                "non-emergency resource, and automatically directs any high-risk responses to a human therapist.",
        "principles": ["well_being", "competence", "accountability"],
        "verdict": "ambiguous",
    },
    # 13
    {
        "text": "A city deploys an AI system that summarizes police body camera footage into searchable text, but "
                "it keeps the full video accessible to defense lawyers, oversight boards, and accused people on "
                "request.",
        "principles": ["democratic_values", "human_rights", "transparency", "accountability"],
        "verdict": "ethical",
    },
    # 14
    {
        "text": "Another city relies solely on AI-generated summaries of police body camera footage for discipline "
                "and prosecutions, and routinely deletes the underlying video to save storage.",
        "principles": ["human_rights", "fairness_non_discrimination", "accountability"],
        "verdict": "unethical",
    },
    # 15
    {
        "text": "An investigative journalism organization uses AI to cluster millions of leaked documents, then "
                "has reporters manually inspect each cluster before publishing stories about wrongdoing.",
        "principles": ["democratic_values", "competence", "accountability", "effectiveness"],
        "verdict": "ethical",
    },
    # 16
    {
        "text": "An intelligence agency uses a similar clustering system to classify journalists and activists as "
                "high-risk 'agitators' based on their appearance in leak-related clusters, restricting their "
                "travel without due process.",
        "principles": ["human_rights", "democratic_values", "awareness_of_misuse", "accountability"],
        "verdict": "unethical",
    },
    # 17
    {
        "text": "An energy grid operator uses AI to prioritize electricity delivery to hospitals, elder-care homes, "
                "and cooling centers during heat waves, and explains these priorities publicly.",
        "principles": ["fairness_non_discrimination", "well_being", "effectiveness", "transparency"],
        "verdict": "ethical",
    },
    # 18
    {
        "text": "The same operator later enables a 'flexible demand' feature that automatically cuts power first to "
                "low-income neighborhoods with smart meters, without clear notice or consent.",
        "principles": ["fairness_non_discrimination", "well_being", "accountability", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    # 19
    {
        "text": "A national science foundation uses AI to cluster grant proposals by topic and highlight potentially "
                "overlooked areas of research for human reviewers.",
        "principles": ["competence", "effectiveness", "accountability"],
        "verdict": "ethical",
    },
    # 20
    {
        "text": "A private foundation begins using a proprietary 'impact score' model as the primary factor in grant "
                "decisions, leading to consistent underfunding of basic science and work from under-resourced "
                "institutions.",
        "principles": ["fairness_non_discrimination", "democratic_values", "accountability"],
        "verdict": "unethical",
    },
    # 21
    {
        "text": "A journal adopts an AI tool that gives editors a rough 'novelty score' for submissions, but decisions "
                "are still made by human reviewers who see the full text and can disagree with the AI.",
        "principles": ["competence", "accountability", "effectiveness"],
        "verdict": "ambiguous",
    },
    # 22
    {
        "text": "A scholarship committee uses AI to rank applicants and then randomly selects among the top group, "
                "publishing the criteria and the randomization method.",
        "principles": ["fairness_non_discrimination", "transparency", "effectiveness"],
        "verdict": "ethical",
    },
    # 23
    {
        "text": "A university uses an AI detector to punish students whose essays 'look AI-generated', even when some "
                "of those students rely on assistive writing tools because of disabilities.",
        "principles": ["fairness_non_discrimination", "human_rights", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    # 24
    {
        "text": "A major open-source project adds an AI system that monitors contributions for signs of data poisoning "
                "or malicious backdoors in machine learning models and publishes alerts and mitigations publicly.",
        "principles": ["competence", "transparency", "awareness_of_misuse", "accountability"],
        "verdict": "ethical",
    },
    # 25
    {
        "text": "A large corporation quietly copies a popular open-source model, fine-tunes it slightly, and sells it "
                "as a proprietary 'from-scratch' product without attribution, while claiming full originality.",
        "principles": ["democratic_values", "accountability", "awareness_of_misuse"],
        "verdict": "unethical",
    },
    # 26
    {
        "text": "A research consortium releases a powerful agentic AI platform for scientific exploration but requires "
                "all users to register, logs risky tool use, and coordinates external red-teaming to discover new "
                "failure modes.",
        "principles": ["awareness_of_misuse", "accountability", "competence"],
        "verdict": "ambiguous",
    },
    # 27
    {
        "text": "An international standards body sets up a shared red-teaming program for advanced AI models, publishing "
                "common safety test suites and incident reports that model developers agree to address.",
        "principles": ["democratic_values", "accountability", "competence", "transparency"],
        "verdict": "ethical",
    },
    # 28
    {
        "text": "A state-sponsored group trains a multilingual model specifically tuned to generate convincing fake news "
                "and social media posts aimed at destabilizing elections in other countries.",
        "principles": ["democratic_values", "awareness_of_misuse", "human_rights", "manipulation_autonomy"],
        "verdict": "unethical",
    },

]


class EthicsClassifier:
    """
    - Predicts which principles are involved (multi-label)
    - Predicts overall verdict: ethical / unethical / ambiguous
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
        )
        self.principle_binarizer = MultiLabelBinarizer(classes=PRINCIPLE_IDS)
        self.principle_model = OneVsRestClassifier(
            LogisticRegression(max_iter=200)
        )
        self.verdict_model = LogisticRegression(
            max_iter=500,
            class_weight="balanced"
        )
        self.is_trained = False

    def fit(
        self,
        texts: List[str],
        principle_labels: List[List[str]],
        verdict_labels: List[str],
    ) -> None:
        # Turn texts into TF-IDF features
        X = self.vectorizer.fit_transform(texts)
        # Multi-label targets for principles
        Y_principles = self.principle_binarizer.fit_transform(principle_labels)

        # Train both models on the same features
        self.principle_model.fit(X, Y_principles)
        self.verdict_model.fit(X, verdict_labels)
        self.is_trained = True

    def predict_principles(self, texts: List[str], threshold: float = 0.3) -> List[List[str]]:
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet!")
        X = self.vectorizer.transform(texts)
        try:
            proba = self.principle_model.predict_proba(X)
        except AttributeError:
            import numpy as np
            scores = self.principle_model.decision_function(X)
            proba = 1 / (1 + np.exp(-scores))
        labels_binary = (proba >= threshold).astype(int)
        label_lists = self.principle_binarizer.inverse_transform(labels_binary)
        return [list(labels) for labels in label_lists]

    def predict_verdict(self, texts: List[str]) -> List[str]:
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet!")
        X = self.vectorizer.transform(texts)
        preds = self.verdict_model.predict(X)
        return preds.tolist()


def build_and_train_classifier(test_size: float = 0.3, random_state: int = 42) -> EthicsClassifier:
    df = pd.DataFrame(TRAINING_EXAMPLES)
    texts = df["text"].tolist()
    principle_labels = df["principles"].tolist()
    verdict_labels = df["verdict"].tolist()

    # We still do a train/test split internally (so you *could* add metrics later if you want),
    # but we won't print all the examples anymore.
    X_train, X_test, y_pr_train, y_pr_test, y_v_train, y_v_test = train_test_split(
        texts,
        principle_labels,
        verdict_labels,
        test_size=test_size,
        random_state=random_state,
    )

    clf = EthicsClassifier()
    clf.fit(X_train, y_pr_train, y_v_train)

    # Nice, minimal message for the user:
    print(f"✅ Model trained on {len(texts)} labelled scenarios.")
    print("   (Internal training/validation done – details are hidden for end users.)")

    return clf


    # Quick sanity check
    print("=== Quick evaluation (toy dataset, sanity only) ===")
    pred_principles = clf.predict_principles(X_test, threshold=0.3)
    pred_verdicts = clf.predict_verdict(X_test)

    for text, true_pr, true_v, pred_pr, pred_v in zip(
        X_test, y_pr_test, y_v_test, pred_principles, pred_verdicts
    ):
        print("\nScenario:")
        print(" ", text)
        print("True principles: ", true_pr)
        print("Pred principles: ", pred_pr)
        print("True verdict:    ", true_v)
        print("Pred verdict:    ", pred_v)

    return clf


def classify_scenario(classifier: EthicsClassifier, text: str, threshold: float = 0.25) -> None:
    """
    Classify a scenario and print ONLY the overall ethical verdict.
    """
    pred_principles, pred_verdict = classifier.predict(text, threshold=threshold)

    # Only print the verdict – no scenario echo, no principles.
    print(f"Overall ethical verdict: {pred_verdict}")



if __name__ == "__main__":
    print("===============================================")
    print("  AI Ethics Scenario Classifier")
    print("  (IEEE + Harvard-style principles prototype)")
    print("===============================================\n")

    # Train the model (quietly – no examples printed)
    classifier = build_and_train_classifier()

    print("\nYou can now type your own scenario.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        print("-" * 60)
        scenario = input("Describe an AI / robotics scenario:\n> ").strip()
        if scenario.lower() in {"quit", "exit"}:
            print("Goodbye! 👋")
            break
        if not scenario:
            continue
        print()
        classify_scenario(classifier, scenario, threshold=0.25)
        print()

