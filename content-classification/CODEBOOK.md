# Corruption-Article Content Codebook

Version: `content_codebook_v1.0.0`

Inputs: `publication_country` and `article_text`.

Assign exactly one label for each variable:

- `victim_visibility`
- `corruption_frame`
- `case_location`
- `accused_actor_visibility`

## General Rules

Use only information stated in the article. Do not use outside knowledge. Code
how the article represents the case, not whether its allegations are true. A
clearly reported allegation counts even when denied, disputed, unproven,
politically motivated, dismissed, or followed by an acquittal.

First isolate the corruption allegation or case. Do not let unrelated crimes,
accidents, violence, repression, personal misconduct, or harmful events affect
the corruption-specific variables.

Use `unclear` only when the text is genuinely ambiguous, incomplete, or
translation-problematic. Do not use it merely because an allegation is unproven
or some details are missing.

`corruption_frame` measures the article's dominant framing.
`accused_actor_visibility` records every type of alleged corruption participant
that is explicitly visible, even when secondary. An article can therefore have
an `individualized` frame and `both_individual_and_organizational` actors.

## 1. Victim Visibility

**Question:** Who or what does the article explicitly represent as having
suffered harm because of the corruption?

### Evidence Gate

Before coding a victim, identify all three elements in the article:

1. a person, group, organization, institution, or public interest;
2. a realized loss, injury, deprivation, or adverse treatment; and
3. a direct connection between that harm and the corruption itself.

If any element is absent or must be supplied by inference, code `no_victim`.
An allegation that harm occurred is sufficient, but merely possible,
hypothetical, intended, or future harm is not.

Do not convert a corrupt act into an assumed consequence. Without an additional
explicit statement of harm, none of the following establishes victimhood:

- public money, property, or an institution is mentioned;
- property is transferred or an illegal project is approved;
- a fictitious contract, payment, or corrupt advantage exists;
- bribery, fraud, embezzlement, money laundering, or tax evasion is reported;
- money is repaid to tax authorities;
- a tender, examination, appointment, contract, or decision favors one actor;
- an investigation, prosecution, conviction, fine, confiscation, resignation,
  scandal, or institutional response occurs;
- corruption creates a risk or is intended to produce harm;
- taxpayers, democracy, society, or public trust would ordinarily be harmed.

Do not infer an unnamed losing bidder, applicant, company, or citizen from
favoritism. The article must identify the deprived party and its loss.

### Causal Source

Count only harm attributed to the corrupt conduct itself. Harm caused by the
accusation, investigation, prosecution, trial, scandal, resignation,
institutional response, political controversy, underfunding, or case-handling
delay does not count unless the article separately attributes it to the
underlying corruption. Harm from an unrelated event in the same article does
not count.

An extortionate demand, corrupt threat, or coercive bribe demand is realized
adverse treatment once communicated. The demanded payment or threatened later
consequence need not occur. Repeated coercive political pressure or interference
directed at a person to stop a corruption investigation also counts when the
article describes that person as subjected to the pressure. Mere lobbying or
political pressure without coercion, interference, or described adverse
treatment does not.

### Labels

`no_victim`

No person, group, organization, institution, or public interest is explicitly
described as suffering corruption-related harm, or the evidence gate is not
satisfied.

`concrete_victim`

An identifiable person, group, community, company, association, or other
bounded non-public entity explicitly suffers direct corruption-related harm.
This includes loss of money, property, rights, services, employment, contracts,
or opportunities; direct exclusion from a manipulated process; extortionate or
coercive pressure; or physical/personal harm explicitly attributed to
corruption. The victim need not be named.

`institutional_societal_victim`

The article explicitly represents corruption as harming a public institution,
public budget or finances, public service, democracy or electoral legitimacy,
rule of law, public trust, institutional credibility or independence, state
capacity, society, social cohesion, the general economy, or development.
Explicit unreimbursed public funds and explicit reputational or institutional
damage to a public institution qualify. A vague statement that an institution
was "hit", "affected", "shaken", criticized, or embarrassed does not qualify
without a specified loss of money, trust, legitimacy, independence, or capacity.

`unclear`

The article may describe corruption-related harm, but incomplete,
contradictory, ambiguous, or translation-problematic wording prevents a
decision.

### Decision Order

1. Identify the corruption allegation.
2. Identify harm caused by that corruption and its victim.
3. If a concrete victim is explicit, code `concrete_victim`.
4. Otherwise, if only broad institutional or societal harm is explicit, code
   `institutional_societal_victim`.
5. If corruption-related harm is not explicit, code `no_victim`.
6. Use `unclear` only when the text prevents a decision.

If concrete and institutional/societal victims are both explicit, select
`concrete_victim`.

### Examples

- "He embezzled EUR 10 million from the company." -> `concrete_victim`
- "Money was collected through racketeering of businesspeople." ->
  `concrete_victim`
- "The scheme undermined Parliament's credibility." ->
  `institutional_societal_victim`
- "Billions in state money were stolen." ->
  `institutional_societal_victim`
- A tender favored friendly companies but no losing party or loss is stated ->
  `no_victim`
- Tax fraud followed by repayment, with no explicit state loss -> `no_victim`
- A corruption proceeding is delayed by staff shortages, reducing service ->
  `no_victim`
- A company is threatened with official retaliation unless it signs a
  fictitious contract -> `concrete_victim`
- Passengers die in an accident and an old corruption charge is mentioned
  separately -> `no_victim`

## 2. Corruption Frame

**Question:** At what level does the article primarily represent the corruption
problem?

Judge the dominant explanation and emphasis. Naming a person does not by itself
make the frame individualized. Mentioning several people, institutions, or
offenses does not by itself make it systemic.

### Labels

`individualized`

Coverage principally centers on identifiable actors and a specific episode:
personal misconduct, accusations, investigation, arrest, trial, conviction,
sentence, resignation, or a particular bribery, fraud, embezzlement, or
conflict-of-interest scandal. Several defendants, an accused company, or brief
systemic background can still form an individualized article.

`systemic`

Coverage principally represents corruption as a broader governance or
institutional pattern: state capture, entrenched clientelism, recurring abuse,
systemic impunity, rule-of-law breakdown, democratic backsliding,
institutionalized protection, or corruption embedded across government,
business, police, courts, or public administration. Named actors may appear,
but the system or recurring pattern must remain central.

`other_or_mixed`

Use when individualized and systemic framing receive roughly equal emphasis;
corruption is incidental or background; the article is mainly procedural,
administrative, legal, or technical; it is a roundup of unrelated cases; it
mainly concerns election-finance regulation or another specialized regulatory
issue; or the article cannot meaningfully be reduced to the
individualized/systemic distinction.

`unclear`

The text is insufficient or unusably ambiguous.

### Decision Order

1. If corruption is incidental, procedural/technical, or evenly mixed, code
   `other_or_mixed`.
2. Otherwise, if institutional dysfunction or a recurring governance pattern
   is the central explanation, code `systemic`.
3. Otherwise, if a specific actor and episode dominate, code `individualized`.
4. Use `unclear` only when the text prevents a decision.

## 3. Case Location

**Question:** Where is the main corruption case relative to
`publication_country`?

Locate the corruption case, not the news agency, quoted speaker, court-reporting
location, or an unrelated main event. Identify where the central actors,
institutions, conduct, or investigation are located and compare that place with
`publication_country`.

### Labels

`domestic`

The principal case concerns the publication country, including domestic actors,
institutions, public contracts, or funds; domestic misconduct involving EU or
foreign money; a domestic actor using offshore accounts; or foreign sanctions
for corruption centered in the publication country. EU funds misused
domestically remain domestic. A case centered in an overseas territory or
constituency belonging to the publication country is also domestic. A
transnational case is domestic when the alleged conduct, investigation, or
central actors are clearly centered in the publication country.

`abroad`

The principal case concerns another country, foreign actors or institutions, a
foreign government or public contract, or a cross-border scheme whose main
center is elsewhere.

`unclear`

No principal location can be determined, including a genuinely transnational
case with several equally central countries and no identifiable center.

## 4. Accused Actor Visibility

**Question:** Which individual and/or organizational actors does the article
explicitly represent as participating in the corruption?

Count only actors whom the article explicitly represents as committing,
attempting, assisting, enabling, financing, directing, or concealing the
corruption. Do not count someone merely because they are a victim, witness,
whistleblower, investigator, prosecutor, judge, regulator, beneficiary,
employer, associate, or owner. Count such an actor only if the article
separately alleges participation in the corruption.

An allegation is sufficient. Conviction is not required, and actors still count
when allegations are denied, charges are dropped, or they are acquitted.

### Mandatory Two-Test Method

**Individual test:** Is there an exact passage accusing a person, officeholder,
or identifiable group of people of participating in the corruption?

**Organization test:** Is there an exact passage in which an organization or
institution is itself accused, investigated, charged, or described as carrying
out, financing, directing, enabling, or concealing corruption?

The organizational allegation may appear in the same sentence as an individual
allegation, but it must be independently expressed. Accusations against an
employee, leader, owner, member, subsidiary, or associate do not automatically
implicate the organization.

An organization does count when the article attributes corrupt participation to
it, for example by stating that it paid bribes, manipulated a tender, financed
a scheme, facilitated money laundering, concealed misconduct, or was
investigated for corruption. A reported allegation that a named party sabotaged
or manipulated an election counts when that conduct is part of the corruption
case. A corporate act explicitly investigated as a corrupt quid pro quo also
counts.

An organization does not count merely because it employed, was led by, was
owned or controlled by, or was associated with an accused person; received a
contract or benefit; appeared in the investigation; hosted the conduct; was a
victim; or was derivatively sanctioned. Mere hiring, a transaction, or a benefit
without an explicit corruption link does not count. Conduct by a leader or
employee becomes organizational conduct only when the article attributes it to
the organization or says the person acted on its behalf.

### Labels

`no_accused_actor`

Corruption is discussed, but neither test identifies a participant. Vague
references such as "they", "certain circles", or "political forces" do not
count without an identifiable actor.

`individual_actor`

At least one person, officeholder, or identifiable group of people is accused,
investigated, charged, convicted, sanctioned for their own conduct, or
explicitly alleged to have participated. Unnamed but clearly identified groups
such as "six provincial deputies" or "former police officers" count. Several
people still satisfy only the individual test.

`organizational_or_institutional_actor`

At least one party, company, ministry, agency, police unit, court, government,
or clearly defined organized group is explicitly alleged to have participated
in its own capacity, and no individual test is satisfied.

`both_individual_and_organizational`

At least one individual and at least one organization independently satisfy
their respective tests.

`unclear`

The wording is too ambiguous to determine whether the alleged participant is a
person, an organization, or neither.

### Mechanical Mapping

- Individual No + Organization No -> `no_accused_actor`
- Individual Yes + Organization No -> `individual_actor`
- Individual No + Organization Yes -> `organizational_or_institutional_actor`
- Individual Yes + Organization Yes -> `both_individual_and_organizational`
- Genuinely ambiguous evidence -> `unclear`

### Examples

- A minister is investigated for accepting bribes; the party is only mentioned
  -> `individual_actor`
- Six unnamed officers are accused of collecting bribes -> `individual_actor`
- A company is owned by a sanctioned politician but is not accused -> do not
  count the company
- A company allegedly paid bribes and its director approved them ->
  `both_individual_and_organizational`
- A party benefited from irregularities but is not accused of organizing them
  -> do not count the party
- A government is accused of concealing a procurement scheme and a minister of
  directing it -> `both_individual_and_organizational`
