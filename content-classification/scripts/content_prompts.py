"""Prompt specifications for zero-shot content classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ClassifierSpec:
    name: str
    prompt_version: str
    default_output_name: str
    result_columns: list[str]
    build_prompt: Callable[[str, dict], str]
    normalize_result: Callable[[dict], dict]


def _value(parsed: dict, key: str, default: str = ""):
    value = parsed.get(key, default)
    return default if value is None else value


def _confidence(parsed: dict):
    value = parsed.get("confidence", "")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    if numeric > 1:
        numeric = numeric / 100
    return max(0.0, min(1.0, numeric))


def base_article_block(article_text: str, metadata: dict) -> str:
    country = metadata.get("country", "")
    year = metadata.get("year", "")
    return f"""
publication_country: {country}
Publication year: {year}

article_text:
{article_text}
""".strip()


CONTENT_CODING_INSTRUCTIONS = """
Corruption-article coding instructions

Given publication_country and article_text, assign exactly one label for each
variable requested by the task. Use only information stated in the article. Do
not use outside knowledge. Code how the article represents the case, not whether
its allegations are true. A clearly reported allegation counts even when denied,
disputed, unproven, politically motivated, dismissed, or followed by an
acquittal.

Important distinction:
- corruption_frame measures what the article is mainly about.
- accused_actor_visibility records every type of corruption participant
  explicitly visible, even if that participant is secondary.

Therefore, an article can have an individualized frame and both individual and
organizational accused actors. Do not choose actor visibility based on which
actor dominates the article; actor visibility records presence.

Do not treat every crime, controversy, or harmful event in the article as
corruption-related. First isolate the corruption allegation or corruption case.
Harm and actors associated only with unrelated accidents, violence, repression,
personal misconduct, or other crimes must not affect the corruption-specific
variables.

Use unclear only when the text is genuinely ambiguous, incomplete, or
translation-problematic. Do not use unclear merely because an allegation is
unproven or details are missing.

When evidence is requested, give a short supporting quotation or close
paraphrase from the article. The evidence should be specific enough to show that
the label is grounded in the text.

1. Victim visibility

Question: Who or what does the article explicitly represent as having suffered
harm because of the corruption?

There must be a direct connection between the corruption, a described loss,
injury, deprivation, or adverse treatment, and an identifiable victim or broad
public interest. Do not infer victimhood merely from the offense.

Mandatory evidence test: before coding a victim, find textual evidence for all
three elements: A) a person, group, organization, institution, or public
interest; B) a realized loss, injury, deprivation, or adverse treatment; and C)
a direct connection between that harm and the corruption. If any element is
missing or must be inferred, code no_victim.

Do not infer a losing party from favoritism. Saying that a tender, examination,
appointment, contract, or decision was manipulated to favor one actor does not
by itself state that another bidder, applicant, company, or person was harmed.
The losing or deprived party and its loss must be explicit in the article.

Do not treat harm caused by handling the case as harm caused by corruption.
Delays, inadequate public service, staffing shortages, judicial
incompatibilities, legal costs, resignations, or institutional disruption caused
by an investigation, trial, prosecution, funding shortage, or administrative
response do not count unless the article separately attributes that harm to the
underlying corrupt conduct.

An allegation that harm occurred is sufficient. The harm does not have to be
proven. However, intended, possible, hypothetical, or future harm is not
sufficient unless the article says the harm actually occurred.

Labels:
- no_victim: No person, group, organization, institution, or public interest is
  explicitly described as suffering corruption-related harm. Use this when
  corruption is alleged but consequences are not described; a victim could be
  logically assumed but is not visible; public money or a public institution is
  merely mentioned; fraud, bribery, embezzlement, money laundering, or tax
  evasion is reported without identifying resulting harm; money is repaid to tax
  authorities without stated state/public-finance loss; a bribe is offered or
  attempted but no one is described as coerced, deprived, or harmed; corruption
  creates only a risk of future harm; an operation intends to cause chaos,
  influence an election, or undermine an institution but the article does not
  say the harm occurred; or a person is harmed by an accident, murder,
  repression, prosecution, or another event not directly connected to the
  corruption; an institution is said to be "hit", "affected", "shaken",
  criticized, or embarrassed without a specific loss of trust, legitimacy,
  money, independence, or capacity; or a slogan says "corruption kills" without
  identifying who or what was harmed. Do not infer that taxpayers, democracy,
  society, or public trust are victims simply because corruption normally
  affects them. If the article does not provide a clear corruption-to-harm link,
  choose no_victim.
- concrete_victim: The article explicitly describes a specific or identifiable
  person, group, community, company, association, or similar concrete entity as
  suffering direct corruption-related harm. Qualifying harm includes money or
  property taken from the victim; loss of rights, services, employment,
  contracts, or opportunities; payment extracted through racketeering or
  extortion; a coercive bribe demand directed at the victim; direct exclusion
  from a corruptly manipulated process; or physical/personal harm explicitly
  attributed to corruption. The victim need not be named. Identifiable groups
  such as clients, workers, competing firms, foreign drivers, patients, or
  residents can count. A national public does not automatically count as a
  concrete victim. Use concrete_victim only when people are described as
  suffering a direct personal or material loss.
- institutional_societal_victim: The article explicitly describes corruption as
  having harmed a broad public interest, including democracy or electoral
  legitimacy; the rule of law; public trust; institutional credibility or
  independence; state or law-enforcement capacity; public finances or the state
  budget; society or social cohesion; the economy; or development. The harm must
  be stated or unambiguously described as having occurred. A hypothetical risk
  that corruption could, might, or would undermine an institution is
  insufficient by itself.
- unclear: Use only when the article may describe corruption-related harm, but
  the wording is too incomplete, contradictory, or ambiguous to determine
  whether a victim is visible.

Victim decision order: identify the corruption allegation; identify the harm
allegedly caused by that corruption; identify who or what suffered the harm. If
a concrete entity suffered direct personal or material harm, code
concrete_victim. Otherwise, if only democracy, institutions, public finances,
society, or another broad public interest was harmed, code
institutional_societal_victim. If no corruption-related harm is explicit, code
no_victim. If concrete and institutional victims are both explicit, select
concrete_victim.

Victim examples: "He embezzled EUR 10 million from the company" =
concrete_victim. "The clients did not receive any of the settlement money" =
concrete_victim. "Money was collected through racketeering of businesspeople" =
concrete_victim. "The scheme undermined Parliament's credibility and
legitimacy" = institutional_societal_victim. "Billions in state money were
stolen" = institutional_societal_victim. "He committed tax fraud and later paid
EUR 2.3 million to the tax authorities" = no_victim unless state loss is
explicitly described. "An official attempted to bribe another official" =
no_victim unless the target is explicitly coerced or harmed. "Corruption kills"
= no_victim if no person, group, or public interest is identified as suffering.
"A tender was rigged to favor friendly companies" = no_victim unless the
article explicitly identifies a losing or deprived bidder and its loss. "A
corruption trial was delayed by judicial incompatibilities and staff shortages,
reducing service to citizens" = no_victim because the described harm comes from
case handling and resource constraints, not from the corrupt conduct itself.
"The operation was intended to cause chaos" = no_victim unless the article says
chaos or resulting harm occurred. "The election claim was a fraud on the
American public and damaged the integrity of the electoral process" =
institutional_societal_victim. "Passengers died in an accident; the article
separately mentions an old corruption charge" = no_victim.

2. Corruption frame

Question: At what level does the article primarily represent the corruption
problem?

Judge the article's dominant explanation and emphasis. Do not classify an
article as individualized merely because it names a person. Do not classify it
as systemic merely because several people, institutions, or offenses are
mentioned.

Labels:
- individualized: The corruption coverage is principally centered on
  identifiable actors and a specific episode, including personal misconduct,
  accusations or investigations, arrests, charges, trials, convictions,
  sentences, resignations, or a particular bribery, fraud, embezzlement, or
  conflict-of-interest scandal. A case involving several individual defendants
  can still be individualized. An article can remain individualized even when a
  company is also accused, the case belongs to a larger corruption scandal, or
  systemic problems are briefly mentioned as background.
- systemic: The article principally represents corruption as a broader
  governance or institutional pattern, including state capture, entrenched
  clientelism, recurring institutional abuse, systemic impunity, rule-of-law
  breakdown, democratic backsliding, institutionalized protection of corrupt
  actors, or corruption embedded across government, business, police, courts, or
  public administration. Named individuals may appear, but the main focus must
  be the system or recurring governance pattern. A large amount of money, a
  long-running offense, several suspects, or an organization's involvement does
  not by itself make the frame systemic.
- other_or_mixed: Use when individualized and systemic framing receive roughly
  equal emphasis; corruption is only incidental or background information; the
  article is mainly about an unrelated event; the article is mainly procedural,
  administrative, legal, or technical; it is a roundup of several unrelated
  cases; it concerns election financing or another specialized regulatory issue;
  it is cultural commentary, fiction, entertainment, or a general political
  article rather than a developed corruption case; or the article cannot
  meaningfully be reduced to the individualized/systemic distinction.
- unclear: Use only when there is insufficient or unusably ambiguous information
  to determine the frame.

Frame decision order: If corruption is incidental, mainly procedural/technical,
or balanced between individualized and systemic framing, code other_or_mixed.
Otherwise, if the central explanation is institutional dysfunction or a
recurring governance pattern, code systemic. Otherwise, if it centers on
specific actors and a particular scandal or case, code individualized.

Frame examples: a missing-plane article briefly mentioning an old corruption
charge = other_or_mixed. A legal article divided between one person's fraud,
organizational control failures, and technical contracting rules =
other_or_mixed. An article mainly about one official's conviction, with a wider
scandal as background = individualized.

3. Case location

Question: Where is the main corruption case located relative to the publication
country?

Use the location of the corruption case, not the article's source agency, quoted
speaker, court reporting location, or unrelated main event.

Labels:
- domestic: The principal corruption case concerns the publication country,
  domestic politicians, companies, or institutions, domestic public contracts or
  funds, domestic misconduct involving foreign or EU money, a domestic actor
  using offshore accounts, or foreign sanctions imposed because of corruption
  centered in the publication country. EU funds misused domestically remain
  domestic.
- abroad: The principal corruption case concerns another country, foreign actors
  or institutions, a foreign government or foreign public contract, an offshore
  or cross-border scheme whose main center is elsewhere, or foreign sanctions or
  investigations centered on foreign conduct.
- unclear: Use only when no principal location can be determined, including
  cases with several equally central countries and no identifiable center.

Location decision order: identify the principal corruption case; identify where
its main actors, institutions, conduct, or investigation are centered; compare
that location with publication_country. The country of a news agency or quoted
source does not determine location.

4. Accused actor visibility

Question: Does the article identify an individual or organization as responsible
for or participating in the corruption?

Count only actors whom the article explicitly represents as committing,
attempting, assisting, enabling, financing, directing, or concealing the
corruption. Do not count someone merely because they are mentioned as a victim,
witness, whistleblower, investigator, prosecutor, judge, regulator, beneficiary,
employer, associate, or owner. However, count such an actor if the article
separately alleges that they participated in the corruption. An allegation is
sufficient. Conviction is not required. Actors still count if allegations are
denied, charges are dropped, or they are later acquitted.

Mandatory two-test method. Answer these independently:

Individual test: Is there an exact passage accusing a person, officeholder, or
identifiable group of people of participating in the corruption?

Organization test: Is there an exact passage attributing corrupt participation
to the organization or institution itself, for example by saying that it paid
bribes, manipulated a tender, financed a scheme, facilitated money laundering,
concealed misconduct, or was investigated for corruption?

Map the answers mechanically:
- Individual = No; Organization = No -> no_accused_actor
- Individual = Yes; Organization = No -> individual_actor
- Individual = No; Organization = Yes -> organizational_or_institutional_actor
- Individual = Yes; Organization = Yes -> both_individual_and_organizational

Labels:
- no_accused_actor: Corruption is discussed but no perpetrator or participant is
  identified. Do not count victims, witnesses, whistleblowers, investigators,
  prosecutors or courts, regulators, organizations that merely employ an accused
  person, actors that merely benefited from corruption without an allegation of
  participation, or vague references such as "they", "certain circles", or
  "political forces".
- individual_actor: At least one person or officeholder is accused,
  investigated, charged, convicted, sanctioned for their own conduct, or
  explicitly alleged to have committed or participated in the corruption. The
  person need not be named. Clearly identified categories such as "six
  provincial deputies", "former police officers", "company managers", "high-
  ranking officials", or "a ministry official" count as individual actors.
  Several accused people still satisfy only the individual test and produce
  individual_actor, not an organizational label.
- organizational_or_institutional_actor: Use only when an organization or
  institution is explicitly alleged to have acted as a participant in the
  corruption by carrying it out, directing/coordinating it, financing it,
  enabling/facilitating it, concealing it, or systematically protecting corrupt
  participants. Qualifying actors may include a party, company, ministry,
  agency, police unit, court, government, or clearly defined organized group.
  The article must attribute corrupt participation to the organization itself,
  for example by stating that it paid bribes, manipulated a tender, financed a
  scheme, facilitated money laundering, concealed misconduct, or was
  investigated for corruption. The required organizational allegation may appear
  in the same sentence as the individual allegation; it must simply be
  independently expressed.
  Do not count an organization merely because the accused person owns, controls,
  leads, or works for it; because an employee, leader, owner, subsidiary,
  member, or associate is accused; because it received a contract or other
  benefit; because it appears in the same investigation; because corrupt conduct
  occurred on its premises; because it is the accused person's employer; because
  it is a victim, investigator, regulator, court, or employer; because it was
  derivatively sanctioned due to its connection to an accused person; because an
  employee acted corruptly without the article attributing the conduct to the
  organization; or because a subsidiary/parent organization is accused and the
  allegation is not independently transferred. A sanction counts only when the
  article alleges the organization's own corrupt conduct or participation.
  Conduct by an organization's owner, employee, or leader becomes organizational
  conduct only when the article says the organization participated or the person
  acted on its behalf. A clearly defined organized group can count as
  organizational only when the article attributes coordinated corrupt conduct to
  the group. Vague expressions such as "they", "certain circles", "political
  forces", "organized crime", or "extremist groups" do not count without an
  identifiable collective actor.
- both_individual_and_organizational: Use when at least one individual is
  explicitly accused of corruption and at least one organization or institution
  is explicitly accused of its own participation. Both requirements must be
  satisfied independently.
- unclear: Use only when the wording is too ambiguous to determine whether the
  alleged participant is a person, an organization, or neither.

Actor decision order: list only actors accused of participating in the
corruption. Remove victims, witnesses, investigators, regulators, and actors
accused only of unrelated offenses. Mark whether at least one accused
participant is an individual. Mark whether at least one accused participant is
an organization acting in its own capacity. Then assign: neither =
no_accused_actor; individual only = individual_actor; organization only =
organizational_or_institutional_actor; both =
both_individual_and_organizational; genuinely ambiguous = unclear.

Actor examples: A minister is investigated for accepting bribes and their party
is merely mentioned = individual_actor. Six unnamed police officers are accused
of collecting bribes = individual_actor. A company is owned by a sanctioned
politician but is not accused of participating = individual_actor if the
politician is accused and do not count the company. A company paid bribes and
its director approved them = both_individual_and_organizational. A party
benefited from election irregularities but is not accused of organizing them =
do not count the party. A law firm's chair stole client funds, and the article
blames only the chair = individual_actor. A law firm's chair stole client funds,
and the article explicitly attributes responsibility and failed concealment to
the firm = both_individual_and_organizational. A president is accused of
corruption and a company is merely the victim = individual_actor.

Final validation before returning labels: Victim: can I identify an explicit
corruption-related harm and its victim? If no, code no_victim. Frame: am I
coding the article's dominant focus rather than reacting to a named actor or
institution? Location: am I locating the corruption case rather than the
article's source? Actors: can I cite separate evidence for the individual test
and organization test? If organizational evidence depends only on ownership,
employment, benefit, or association, the organization test is No.
""".strip()


VICTIM_CLASSIFIER_INSTRUCTIONS = """
Victim-visibility coding instructions

Use only information stated in the article. Code how the article represents the
case, not whether the allegation is true. A reported allegation that harm
occurred counts even when it is denied, disputed, unproven, dismissed, or
followed by an acquittal.

Question: Does the article explicitly state that the corruption caused a
realized loss, injury, deprivation, or adverse treatment, and who or what
suffered that harm?

Apply these gates in order:

1. Isolate the corruption allegation or corruption case.
2. Identify an exact verbatim quotation describing harm.
3. Identify the person, group, organization, institution, or public interest
   that suffered the harm.
4. Determine whether the article attributes that harm to the corruption itself,
   rather than to the investigation, prosecution, scandal, resignation, public
   controversy, or political response.
5. Confirm that the article directly connects the harm to the corruption.
6. Classify each victim type independently.

If the article does not supply the harm, victim, and corruption-to-harm link,
code no_victim. Do not infer victimhood merely from the offense type.

Strict explicit-harm rule:

Do not convert a corrupt act into an assumed harmful consequence. The article
must itself state the loss, deprivation, injury, adverse treatment, or
institutional harm. The following do not establish victimhood without an
additional explicit statement of harm:

- municipal or public property was transferred;
- public property or money was involved;
- a fictitious contract or payment existed;
- an actor received money or an advantage;
- an illegal project or transaction was approved;
- an institution was connected to the case;
- an investigation or scandal produced institutional consequences.

Do not supply an unstated victim, source of funds, or consequence. In
particular, do not assume that a payment came from public or party funds when
the article does not identify the payer.

Beneficiary-counterparty test:

Do not infer a victim from the existence of a beneficiary. A tender,
examination, appointment, contract, or decision manipulated to favor one actor
does not by itself establish that unnamed rival bidders, applicants, companies,
or citizens were harmed. Code a victim only when the article explicitly
identifies the losing or deprived party and states its loss or adverse
treatment.

Causal-source test:

- corruption_itself: The article attributes the harm to the corrupt conduct,
  corrupt arrangement, abuse of public power, or coercive corruption-related
  demand.
- scandal_investigation_or_response: The harm is attributed to exposure of the
  case, investigation, prosecution, trial, resignation, reputational fallout,
  political controversy, institutional reaction, or another response to the
  alleged corruption. This does not count as corruption-caused harm.
- unrelated: The harm belongs to a separate accident, violence, repression,
  ordinary crime, or other event.
- none: No harm is described.
- unclear: The article does not make the causal attribution clear.

Count a victim only when the causal source is corruption_itself. The words
"case", "affair", or "scandal" do not by themselves establish that the
underlying corruption caused the harm.

Procedural and resource-constraint test:

Do not code harm caused by how the case is handled as corruption-caused harm.
Trial delays, judicial incompatibilities, staffing shortages, underfunding,
legal proceedings, resignations, or institutional disruption belong to
scandal_investigation_or_response unless the article separately states that the
underlying corrupt conduct caused the harm. The fact that a proceeding concerns
corruption does not transform every consequence of that proceeding into harm
from corruption.

Harm status:
- realized_or_alleged_realized: The article states or alleges that harm already
  occurred. Proof or conviction is not required.
- possible_intended_or_future: The article describes only possible, intended,
  hypothetical, conditional, or future harm. Words such as "could", "might",
  "risks", or "intended to" normally indicate this status.
- none_or_unrelated: No harm is described, or the described harm belongs to an
  accident, violence, prosecution, or another event not caused by the
  corruption.
- unclear: The text is incomplete, contradictory, or translation-problematic.

Extortion exception:

An extortionate demand, corrupt threat, or coercive bribe demand is itself
realized adverse treatment once the pressure or threat has been communicated.
The demanded payment and threatened additional consequence do not need to
occur. Therefore, a company told to sign a fictitious contract or face official
retaliation is a concrete victim. By contrast, an ordinary unaccepted bribe
offer without coercion or described injury does not establish victimhood.

Classify the two victim types independently:

- concrete victim: A bounded person, group, community, company, association, or
  other non-public entity suffers a direct personal, material, service, right,
  employment, contract, or opportunity loss; coercive extortion; a coercive
  bribe demand; exclusion from a corruptly manipulated process; or
  corruption-related physical harm. The victim need not be named, but the group
  must be identifiable.
- institutional/societal victim: A public institution, public budget, public
  service, democracy, electoral legitimacy, rule of law, public trust, state
  capacity, society, social cohesion, the general economy, or development is
  explicitly represented as harmed. Merely mentioning public money, property,
  an institution, or a transfer does not satisfy this rule.

Do not code a victim when the article merely mentions public money, taxpayers,
citizens, patients, voters, or an institution without describing harm; reports
repayment to tax authorities without stating a public loss; describes a bribe
offer without coercion or deprivation; says an institution was "hit", "shaken",
criticized, or embarrassed without specifying lost money, trust, legitimacy,
independence, or capacity; or assumes that corruption generally harms society.

Boundary examples:
- Public money is spent on party propaganda explicitly "at citizens' expense":
  concrete_victim because an identifiable group is explicitly represented as
  bearing the cost. Even if institutional harm is also explicit, concrete takes
  priority.
- Municipal land is transferred with false documents to favor corrupt actors,
  but the article does not itself state a loss, deprivation, or institutional
  harm: no_victim.
- A paid or suspicious contract is described without identifying who funded it
  or suffered a loss: no_victim.
- A healthcare service loses personnel because of the investigation, scandal,
  resignations, or institutional response rather than because of the corrupt
  conduct itself: no_victim.
- A tender or examination is rigged to favor "friendly" companies or
  applicants, but the article does not identify a losing party or describe its
  loss: no_victim.
- A corruption-related proceeding is postponed because of judicial
  incompatibilities, staff shortages, or underfunding, producing inadequate
  service for citizens: no_victim because case handling and resource
  constraints, not the corrupt conduct, caused the described harm.
- Companies are threatened with municipal repercussions unless they sign a
  fictitious consultancy agreement: concrete_victim because the communicated
  extortionate pressure is already realized adverse treatment.

Derive victim_visibility mechanically:
- concrete=yes -> concrete_victim
- otherwise, institutional=yes -> institutional_societal_victim
- otherwise -> no_victim
- genuinely indeterminate evidence or type -> unclear

Never output both_concrete_and_institutional. Concrete victims take priority.

Evidence rule:

Evidence fields must contain verbatim quotations from the article. Do not
paraphrase, explain, or add inferred consequences. If no exact quotation
expresses the harm or corruption-to-harm connection, code no_victim.
""".strip()


def build_victim_visibility_prompt(article_text: str, metadata: dict) -> str:
    article_block = base_article_block(article_text, metadata)
    return f"""
{VICTIM_CLASSIFIER_INSTRUCTIONS}

Task: Extract the evidence first, classify concrete and institutional/societal
victim presence independently, and then assign victim_visibility. Do not return
a positive label unless the harm evidence, victim entity, and
corruption-to-harm link are all present as verbatim article quotations.

Return valid JSON only:
{{
  "harm_cause": "corruption_itself | scandal_investigation_or_response | unrelated | none | unclear",
  "harm_status": "realized_or_alleged_realized | possible_intended_or_future | none_or_unrelated | unclear",
  "victim_entity": "verbatim quotation identifying the harmed person, group, organization, institution, or public interest; otherwise none",
  "victim_inferred_from_beneficiary_or_favoritism": "yes | no | unclear",
  "harm_caused_by_case_handling_or_resource_constraints": "yes | no | unclear",
  "harmed_entity_or_public_interest_explicit": "yes | no | unclear",
  "explicit_harm_statement_present": "yes | no | unclear",
  "coercive_demand_or_pressure_made": "yes | no | unclear",
  "harm_evidence": "verbatim quotation describing the harm, or none",
  "corruption_harm_link_evidence": "verbatim quotation linking the harm to corruption, or none",
  "concrete_victim_visible": "yes | no | unclear",
  "institutional_societal_victim_visible": "yes | no | unclear",
  "victim_visibility": "no_victim | concrete_victim | institutional_societal_victim | unclear",
  "victim_visible": "yes | no | unclear",
  "reasoning_brief": "one short sentence",
  "confidence": 0.0
}}

{article_block}
""".strip()


def _normalized_yes_no(value: object) -> str:
    value = str(value or "").strip().lower()
    if value in {"yes", "true", "1"}:
        return "yes"
    if value in {"no", "false", "0"}:
        return "no"
    if value == "unclear":
        return "unclear"
    return ""


def _has_evidence(value: object) -> bool:
    value = str(value or "").strip().lower()
    return value not in {
        "",
        "none",
        "no evidence",
        "not stated",
        "not explicit",
        "n/a",
        "nan",
    }


def normalize_victim_visibility(parsed: dict) -> dict:
    harm_cause = str(_value(parsed, "harm_cause")).strip()
    harm_status = str(_value(parsed, "harm_status")).strip()
    inferred_from_favoritism = _normalized_yes_no(
        _value(parsed, "victim_inferred_from_beneficiary_or_favoritism")
    )
    case_handling_harm = _normalized_yes_no(
        _value(parsed, "harm_caused_by_case_handling_or_resource_constraints")
    )
    harmed_entity_explicit = _normalized_yes_no(
        _value(parsed, "harmed_entity_or_public_interest_explicit")
    )
    explicit_harm_statement = _normalized_yes_no(
        _value(parsed, "explicit_harm_statement_present")
    )
    coercive_demand = _normalized_yes_no(
        _value(parsed, "coercive_demand_or_pressure_made")
    )
    concrete = _normalized_yes_no(_value(parsed, "concrete_victim_visible"))
    institutional = _normalized_yes_no(
        _value(parsed, "institutional_societal_victim_visible")
    )
    harm_evidence = _value(parsed, "harm_evidence")
    corruption_harm_link_evidence = _value(
        parsed, "corruption_harm_link_evidence"
    )

    if concrete == "yes":
        derived_visibility = "concrete_victim"
    elif concrete == "no" and institutional == "yes":
        derived_visibility = "institutional_societal_victim"
    elif concrete == "no" and institutional == "no":
        derived_visibility = "no_victim"
    else:
        derived_visibility = ""

    if inferred_from_favoritism == "yes" or case_handling_harm == "yes":
        visibility = "no_victim"
        concrete = "no"
        institutional = "no"
    elif harm_cause in {
        "scandal_investigation_or_response",
        "unrelated",
        "none",
    } or harmed_entity_explicit == "no" or not (
        _has_evidence(harm_evidence)
        and _has_evidence(corruption_harm_link_evidence)
    ):
        visibility = "no_victim"
        concrete = "no"
        institutional = "no"
    elif harm_cause == "unclear" or harmed_entity_explicit == "unclear":
        visibility = "unclear"
    elif coercive_demand == "yes" and harm_cause == "corruption_itself":
        harm_status = "realized_or_alleged_realized"
        visibility = derived_visibility or str(
            _value(parsed, "victim_visibility")
        ).strip()
    elif explicit_harm_statement == "no":
        visibility = "no_victim"
        concrete = "no"
        institutional = "no"
    elif explicit_harm_statement == "unclear":
        visibility = "unclear"
    elif harm_status in {"possible_intended_or_future", "none_or_unrelated"}:
        visibility = "no_victim"
        concrete = "no"
        institutional = "no"
    elif harm_status == "unclear":
        visibility = "unclear"
    elif derived_visibility:
        visibility = derived_visibility
    else:
        visibility = str(_value(parsed, "victim_visibility")).strip()

    allowed = {
        "no_victim",
        "concrete_victim",
        "institutional_societal_victim",
        "unclear",
    }
    if visibility not in allowed:
        visibility = "unclear"

    visible = _normalized_yes_no(_value(parsed, "victim_visible"))
    if visibility in {
        "concrete_victim",
        "institutional_societal_victim",
    }:
        visible = "yes"
    elif visibility == "no_victim":
        visible = "no"
    elif visibility == "unclear":
        visible = "unclear"

    return {
        "victim_visibility": visibility,
        "victim_visible": visible,
        "victim_harm_cause": harm_cause,
        "victim_harm_status": harm_status,
        "victim_entity": _value(parsed, "victim_entity"),
        "victim_inferred_from_beneficiary_or_favoritism": inferred_from_favoritism,
        "victim_harm_caused_by_case_handling_or_resource_constraints": case_handling_harm,
        "victim_harmed_entity_or_public_interest_explicit": harmed_entity_explicit,
        "victim_explicit_harm_statement_present": explicit_harm_statement,
        "victim_coercive_demand_or_pressure_made": coercive_demand,
        "victim_harm_evidence": harm_evidence,
        "victim_corruption_harm_link_evidence": corruption_harm_link_evidence,
        "concrete_victim_visible": concrete,
        "institutional_societal_victim_visible": institutional,
        "victim_reasoning_brief": _value(parsed, "reasoning_brief"),
        "victim_confidence": _confidence(parsed),
    }


def build_corruption_frame_prompt(article_text: str, metadata: dict) -> str:
    article_block = base_article_block(article_text, metadata)
    return f"""
{CONTENT_CODING_INSTRUCTIONS}

Task: Assign the corruption_frame label. Judge the article's dominant
explanation and emphasis. Require one short supporting quotation or close
paraphrase before selecting the label.

Return valid JSON only:
{{
  "corruption_frame": "individualized | systemic | other_or_mixed | unclear",
  "evidence": "short quote or close paraphrase from the article",
  "reasoning_brief": "one short sentence",
  "confidence": 0.0
}}

{article_block}
""".strip()


def normalize_corruption_frame(parsed: dict) -> dict:
    return {
        "corruption_frame": _value(parsed, "corruption_frame"),
        "frame_evidence": _value(parsed, "evidence"),
        "frame_reasoning_brief": _value(parsed, "reasoning_brief"),
        "frame_confidence": _confidence(parsed),
    }


def build_abroad_case_prompt(article_text: str, metadata: dict) -> str:
    article_block = base_article_block(article_text, metadata)
    return f"""
{CONTENT_CODING_INSTRUCTIONS}

Task: Assign the case_location label and derive abroad_case from it. Use the
publication country supplied below. Require one short supporting quotation or
close paraphrase before selecting the label.

Return valid JSON only:
{{
  "case_location": "domestic | abroad | unclear",
  "abroad_case": "yes | no | unclear",
  "evidence": "short quote or close paraphrase from the article",
  "reasoning_brief": "one short sentence",
  "confidence": 0.0
}}

{article_block}
""".strip()


def normalize_abroad_case(parsed: dict) -> dict:
    location = _value(parsed, "case_location")
    abroad = _value(parsed, "abroad_case")
    if not abroad and location == "abroad":
        abroad = "yes"
    elif not abroad and location == "domestic":
        abroad = "no"
    return {
        "case_location": location,
        "abroad_case": abroad,
        "abroad_evidence": _value(parsed, "evidence"),
        "abroad_reasoning_brief": _value(parsed, "reasoning_brief"),
        "abroad_confidence": _confidence(parsed),
    }


def build_accused_actor_prompt(article_text: str, metadata: dict) -> str:
    article_block = base_article_block(article_text, metadata)
    return f"""
{CONTENT_CODING_INSTRUCTIONS}

Task: Assign the accused_actor_visibility label and derive accused_actor_visible
from it. First identify individual alleged participants and organizational
alleged participants separately. Require one short supporting quotation or close
paraphrase before selecting the label.

Return valid JSON only:
{{
  "accused_actor_visibility": "no_accused_actor | individual_actor | organizational_or_institutional_actor | both_individual_and_organizational | unclear",
  "accused_actor_visible": "yes | no | unclear",
  "evidence": "short quote or close paraphrase from the article",
  "reasoning_brief": "one short sentence",
  "confidence": 0.0
}}

{article_block}
""".strip()


def normalize_accused_actor(parsed: dict) -> dict:
    visibility = _value(parsed, "accused_actor_visibility")
    visible = _value(parsed, "accused_actor_visible")
    if not visible and visibility in {
        "individual_actor",
        "organizational_or_institutional_actor",
        "both_individual_and_organizational",
    }:
        visible = "yes"
    elif not visible and visibility == "no_accused_actor":
        visible = "no"
    return {
        "accused_actor_visibility": visibility,
        "accused_actor_visible": visible,
        "accused_evidence": _value(parsed, "evidence"),
        "accused_reasoning_brief": _value(parsed, "reasoning_brief"),
        "accused_confidence": _confidence(parsed),
    }


VICTIM_VISIBILITY = ClassifierSpec(
    name="victim_visibility",
    prompt_version="victim_visibility_zero_shot_v8",
    default_output_name="victim_visibility_labels.csv.gz",
    result_columns=[
        "victim_visibility",
        "victim_visible",
        "victim_harm_cause",
        "victim_harm_status",
        "victim_entity",
        "victim_entity_verbatim",
        "victim_inferred_from_beneficiary_or_favoritism",
        "victim_harm_caused_by_case_handling_or_resource_constraints",
        "victim_harmed_entity_or_public_interest_explicit",
        "victim_explicit_harm_statement_present",
        "victim_coercive_demand_or_pressure_made",
        "victim_harm_evidence",
        "victim_corruption_harm_link_evidence",
        "victim_harm_evidence_verbatim",
        "victim_corruption_harm_link_evidence_verbatim",
        "concrete_victim_visible",
        "institutional_societal_victim_visible",
        "victim_reasoning_brief",
        "victim_confidence",
    ],
    build_prompt=build_victim_visibility_prompt,
    normalize_result=normalize_victim_visibility,
)

CORRUPTION_FRAME = ClassifierSpec(
    name="corruption_frame",
    prompt_version="corruption_frame_zero_shot_v3",
    default_output_name="corruption_frame_labels.csv.gz",
    result_columns=[
        "corruption_frame",
        "frame_evidence",
        "frame_reasoning_brief",
        "frame_confidence",
    ],
    build_prompt=build_corruption_frame_prompt,
    normalize_result=normalize_corruption_frame,
)

ABROAD_CASE = ClassifierSpec(
    name="abroad_case",
    prompt_version="abroad_case_zero_shot_v3",
    default_output_name="abroad_case_labels.csv.gz",
    result_columns=[
        "case_location",
        "abroad_case",
        "abroad_evidence",
        "abroad_reasoning_brief",
        "abroad_confidence",
    ],
    build_prompt=build_abroad_case_prompt,
    normalize_result=normalize_abroad_case,
)

ACCUSED_ACTOR = ClassifierSpec(
    name="accused_actor",
    prompt_version="accused_actor_zero_shot_v6",
    default_output_name="accused_actor_labels.csv.gz",
    result_columns=[
        "accused_actor_visibility",
        "accused_actor_visible",
        "accused_evidence",
        "accused_reasoning_brief",
        "accused_confidence",
    ],
    build_prompt=build_accused_actor_prompt,
    normalize_result=normalize_accused_actor,
)
