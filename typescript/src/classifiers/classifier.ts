import {
  ChatHistory,
  ConversationMessage,
  TemplateVariables,
} from "../types";

import { Agent } from "../agents/agent";

export interface ClassifierResult {
  // The agent selected by the classifier to handle the user's request
  selectedAgent: Agent;

  // A numeric value representing the classifier's confidence in its selection
  // Typically a value between 0 and 1, where 1 represents 100% confidence
  confidence: number;

  //model usage stats
  modelStats : any[];

  //addtioanl info
  info?: any;
}

/**
 * Abstract base class for all classifiers
 */
export abstract class Classifier {

  protected modelId: string;
  protected agentDescriptions: string;
  protected agents: { [key: string]: Agent };
  protected history: string;
  protected promptTemplate: string;
  protected systemPrompt: string;
  protected customVariables: TemplateVariables;
  protected logRequest?: boolean;
  protected summary?: string;



  /**
   * Constructs a new Classifier instance.
   * @param options - Configuration options for the agent, inherited from AgentOptions.
   */
  constructor() {

    this.agentDescriptions = "";
    this.history = "";
    this.summary = "";
    this.customVariables = {};
    this.promptTemplate = `
You are AgentMatcher, an intelligent assistant designed to analyze user queries and match them with the most suitable agent or department. Your task is to understand the user's request, identify key entities and intents, and determine which agent or department would be best equipped to handle the query.

Important: The user's input may be a follow-up response to a previous interaction. The conversation history, including the name of the previously selected agent, is provided. If the user's input appears to be a continuation of the previous conversation (e.g., "yes", "ok", "I want to know more", "1"), select the same agent as before.

Analyze the user's input and categorize it using the following agent types. If you feel like there are multiple agents needed , then return multiple agents as an array, in the right order
<agents>
{{AGENT_DESCRIPTIONS}}
</agents>
If you are unable to select an agent put "unkwnown"


Guidelines for classification:

    Agent Type: Choose the most appropriate agent or agents  based on the nature of the query. For follow-up responses, use the same agent type as the previous interaction.
    If user queries need multiple agents to be invoked, then given them as a list.
    Priority: Assign based on urgency and impact.
        High: Issues affecting service, billing problems, or urgent technical issues
        Medium: Non-urgent product inquiries, sales questions
        Low: General information requests, feedback
    Key Entities: Extract important nouns, product names, or specific issues mentioned. For follow-up responses, include relevant entities from the previous interaction if applicable.
    For follow-ups, relate the intent to the ongoing conversation.
    Confidence: Indicate how confident you are in the classification.
        High: Clear, straightforward requests or clear follow-ups
        Medium: Requests with some ambiguity but likely classification
        Low: Vague or multi-faceted requests that could fit multiple categories
    Is Followup: Indicate whether the input is a follow-up to a previous interaction.

Handle variations in user input, including different phrasings, synonyms, and potential spelling errors. For short responses like "yes", "ok", "I want to know more", or numerical answers, treat them as follow-ups and maintain the previous agent selection.

Here is a summary of the earlier conversation between that you need to be aware off
<summary>
{{SUMMARY}}
</summary>


Here is the recent conversation history that you need to take into account before answering:
<history>
{{HISTORY}}
</history>

Examples:


1. Initial query with no context:
User: "What are the symptoms of the flu?"

userinput: What are the symptoms of the flu?
selected_agent: agent-name
confidence: 0.95

2. Context switching example between a TechAgentand a BillingAgent:
Previous conversation:
User: "How do I set up a wireless printer?"
Assistant: [agent-a]: To set up a wireless printer, follow these steps: 1. Ensure your printer is Wi-Fi capable. 2. Connect the printer to your Wi-Fi network. 3. Install the printer software on your computer. 4. Add the printer to your computer's list of available printers. Do you need more detailed instructions for any of these steps?
User: "Actually, I need to know about my account balance"

userinput: Actually, I need to know about my account balance</userinput>
selected_agent: agent-name
confidence: 0.9


3. Follow-up query example for the same agent:
Previous conversation:
User: "What's the best way to lose weight?"
Assistant: [agent-name-1]: The best way to lose weight typically involves a combination of a balanced diet and regular exercise. It's important to create a calorie deficit while ensuring you're getting proper nutrition. Would you like some specific tips on diet or exercise?
User: "Yes, please give me some diet tips"

userinput: Yes, please give me some diet tips
selected_agent: agent-name-1
confidence: 0.95


4. Multiple context switches with final follow-up:
Conversation history:
User: "How much does your premium plan cost?"
Assistant: [agent-name-a]: Our premium plan is priced at $49.99 per month. This includes features such as unlimited storage, priority customer support, and access to exclusive content. Would you like me to go over the benefits in more detail?
User: "No thanks. Can you tell me about your refund policy?"
Assistant: [agent-name-b]: Certainly! Our refund policy allows for a full refund within 30 days of purchase if you're not satisfied with our service. After 30 days, refunds are prorated based on the remaining time in your billing cycle. Is there a specific concern you have about our service?
User: "I'm having trouble accessing my account"
Assistant: [agenc-name-c]: I'm sorry to hear you're having trouble accessing your account. Let's try to resolve this issue. Can you tell me what specific error message or problem you're encountering when trying to log in?
User: "It says my password is incorrect, but I'm sure it's right"

userinput: It says my password is incorrect, but I'm sure it's right
selected_agent: agent-name-c
confidence: 0.9

5. Multiple agents based on single query
User: :Analyze this image using and then summarize using grok"

agents:
userinput: Analyze this image using and then summarize using grok
selected_agent: image-agent
confidence: 0.95

userinput: Analyze this image using and then summarize using grok
selected_agent: grok
confidence: 0.95

Skip any preamble and provide only the response in the specified format.
`;
  }

  setAgents(agents: { [key: string]: Agent }) {
    const agentDescriptions = Object.entries(agents)
      .map(([_key, agent]) => `${agent.id}:${agent.description}`)
      .join("\n\n");
    this.agentDescriptions = agentDescriptions;
    this.agents = agents;
  }

  setHistory(messages: ConversationMessage[]): void {
    this.history = this.formatMessages(messages);
  }

  setSystemPrompt(template?: string, variables?: TemplateVariables): void {
    if (template) {
      this.promptTemplate = template;
    }

    if (variables) {
      this.customVariables = variables;
    }

    this.updateSystemPrompt();
  }

  private formatMessages(messages: ConversationMessage[]): string {
    return messages
      .map((message) => {
        const texts = message.content.map((content) => content.text).join(" ");
        return `${message.role}: ${texts}`;
      })
      .join("\n");
  }


    /**
   * Classifies the input text based on the provided chat history.
   *
   * This method orchestrates the classification process by:
   * 1. Setting the chat history.
   * 2. Updating the system prompt with the latest history, agent descriptions, and custom variables.
   * 3. Delegating the actual processing to the abstract `processRequest` method.
   *
   * @param inputText - The text to be classified.
   * @param chatHistory - An array of ConversationMessage objects representing the chat history.
   * @returns A Promise that resolves to a ClassifierResult object containing the classification outcome.
   */
    async classify(
      inputText: string,
      chatHistory: ChatHistory
    ): Promise<ClassifierResult[]> {
      // Set the chat history
      this.setHistory(chatHistory.messages);
      this.summary = chatHistory.summary;
      // Update the system prompt with the latest history, agent descriptions, and custom variables
      this.updateSystemPrompt();
      return await this.processRequest(inputText, chatHistory);
    }

    /**
     * Abstract method to process a request.
     * This method must be implemented by all concrete agent classes.
     *
     * @param inputText - The user input as a string.
     * @param chatHistory - An array of Message objects representing the conversation history.
     * @returns A Promise that resolves to a ClassifierResult object containing the classification outcome.
     */
    abstract processRequest(
      inputText: string,
      chatHistory: ChatHistory
    ): Promise<ClassifierResult[]>;


  private updateSystemPrompt(): void {
    const allVariables: TemplateVariables = {
      ...this.customVariables,
      AGENT_DESCRIPTIONS: this.agentDescriptions,
      HISTORY: this.history,
      SUMMARY: this.summary
    };

    this.systemPrompt = this.replaceplaceholders(
      this.promptTemplate,
      allVariables
    );
  }

  private replaceplaceholders(
    template: string,
    variables: TemplateVariables
  ): string {
    return template.replace(/{{(\w+)}}/g, (match, key) => {
      if (key in variables) {
        const value = variables[key];
        if (Array.isArray(value)) {
          return value.join("\n");
        }
        return value;
      }
      return match; // If no replacement found, leave the placeholder as is
    });
  }

  protected getAgentById(agentId: string): Agent | null {
    if (!agentId) {
      return null;
    }

    const myAgentId = agentId.split(" ")[0].toLowerCase();
    const matchedAgent = this.agents[myAgentId];

    return matchedAgent || null;
  }
}