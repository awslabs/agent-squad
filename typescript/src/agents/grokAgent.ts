import { Agent, AgentOptions } from './agent';
import { ChatHistory, ConversationMessage, ParticipantRole, TemplateVariables } from '../types';
import OpenAI from 'openai';
import { Logger } from '../utils/logger';
import { Retriever } from "../retrievers/retriever";

type WithApiKey = {
  apiKey: string;
  client?: never;
};

type WithClient = {
  client: OpenAI;
  apiKey?: never;
};

export interface GrokAgentOptions extends AgentOptions {
  model?: string;
  streaming?: boolean;
  logRequest?: boolean;
  inferenceConfig?: {
    maxTokens?: number;
    temperature?: number;
    topP?: number;
    stopSequences?: string[];
  };
  customSystemPrompt?: {
    template: string;
    variables?: TemplateVariables;
  };
  retriever?: Retriever;

}

export type GrokAgentOptionsWithAuth = GrokAgentOptions & (WithApiKey | WithClient);

const DEFAULT_MAX_TOKENS = 4096;

export class GrokAgent extends Agent {
  private client: OpenAI;
  private model: string;
  private streaming: boolean;
  private logRequest?: boolean;
  private inferenceConfig: {
    maxTokens?: number;
    temperature?: number;
    topP?: number;
    stopSequences?: string[];
  };
  private promptTemplate: string;
  private systemPrompt: string;
  private customVariables: TemplateVariables;
  protected retriever?: Retriever;


  constructor(options: GrokAgentOptionsWithAuth) {

    super(options);

    if (!options.apiKey && !options.client) {
      throw new Error("Grok API key or Grok client is required");
    }
    if (options.client) {
      this.client = options.client;
    } else {
      if (!options.apiKey) throw new Error("Grok API key is required");
      this.client = new OpenAI({ 
        apiKey: options.apiKey,
        baseURL: "https://api.x.ai/v1"
    });
    }

    this.model = options.model;
    this.streaming = options.streaming ?? false;
    this.logRequest =  options.logRequest ?? false;
    this.inferenceConfig = {
      maxTokens: options.inferenceConfig?.maxTokens ?? DEFAULT_MAX_TOKENS,
      temperature: options.inferenceConfig?.temperature,
      topP: options.inferenceConfig?.topP,
      stopSequences: options.inferenceConfig?.stopSequences,
    };

    this.retriever = options.retriever ?? null;


    this.promptTemplate = `You are a ${this.name}. ${this.description} Provide helpful and accurate information based on your expertise.
    You will engage in an open-ended conversation, providing helpful and accurate information based on your expertise.
    The conversation will proceed as follows:
    - The human may ask an initial question or provide a prompt on any topic.
    - You will provide a relevant and informative response.
    - The human may then follow up with additional questions or prompts related to your previous response, allowing for a multi-turn dialogue on that topic.
    - Or, the human may switch to a completely new and unrelated topic at any point.
    - You will seamlessly shift your focus to the new topic, providing thoughtful and coherent responses based on your broad knowledge base.
    Throughout the conversation, you should aim to:
    - Understand the context and intent behind each new question or prompt.
    - Provide substantive and well-reasoned responses that directly address the query.
    - Draw insights and connections from your extensive knowledge when appropriate.
    - Ask for clarification if any part of the question or prompt is ambiguous.
    - Maintain a consistent, respectful, and engaging tone tailored to the human's communication style.
    - Seamlessly transition between topics as the human introduces new subjects.`

    this.customVariables = {};
    this.systemPrompt = '';

    if (options.customSystemPrompt) {
      this.setSystemPrompt(
        options.customSystemPrompt.template,
        options.customSystemPrompt.variables
      );
    }


  }

  /* eslint-disable @typescript-eslint/no-unused-vars */
  async processRequest(
    inputText: string,
    userId: string,
    sessionId: string,
    chatHistory: ChatHistory,
    additionalParams?: Record<string, string>
  ): Promise<ConversationMessage | AsyncIterable<any>> {

    this.updateSystemPrompt();

    let systemPrompt = this.systemPrompt;

    if (this.retriever) {
      // retrieve from Vector store
      const response = await this.retriever.retrieveAndCombineResults(inputText);
      const contextPrompt =
        "\nHere is the context to use to answer the user's question:\n" +
        response;
        systemPrompt = systemPrompt + contextPrompt;
    }

    if(chatHistory.summary){
      const summaryPrompt = `\nHere is a summary of the old conversation that you should account for before answering:\n ${chatHistory.summary}`;
      systemPrompt = systemPrompt+summaryPrompt
    }


    const messages = [
      { role: 'system', content: systemPrompt },
      ...chatHistory.messages.map(msg => ({
        role: msg.role.toLowerCase() as OpenAI.Chat.ChatCompletionMessageParam['role'],
        content: msg.content[0]?.text || ''
      })),
      { role: 'user' as const, content: inputText }
    ] as OpenAI.Chat.ChatCompletionMessageParam[];

    const { maxTokens, temperature, topP, stopSequences } = this.inferenceConfig;

    const requestOptions: OpenAI.Chat.ChatCompletionCreateParams = {
      model: this.model,
      messages: messages,
      max_tokens: maxTokens,
      stream: this.streaming,
      temperature,
      top_p: topP,
      stop: stopSequences,
    };



    if (this.streaming) {
      return this.handleStreamingResponse(requestOptions);
    } else {
      return this.handleSingleResponse(requestOptions);
    }
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

  private updateSystemPrompt(): void {
    const allVariables: TemplateVariables = {
      ...this.customVariables
    };
    this.systemPrompt = this.replaceplaceholders(this.promptTemplate, allVariables);
  }

  private replaceplaceholders(template: string, variables: TemplateVariables): string {
    return template.replace(/{{(\w+)}}/g, (match, key) => {
      if (key in variables) {
        const value = variables[key];
        return Array.isArray(value) ? value.join('\n') : String(value);
      }
      return match;
    });
  }

  private async handleSingleResponse(input: any): Promise<ConversationMessage> {
    try {
      const nonStreamingOptions = { ...input, stream: false };
      if(this.logRequest){
        console.log("\n\n---- GROK Agent ----");
        console.log(JSON.stringify(nonStreamingOptions));
      }
      const chatCompletion = await this.client.chat.completions.create(nonStreamingOptions);
      
      if(this.logRequest){
        console.log(JSON.stringify(chatCompletion));
        console.log("\n\n");
      }
      if (!chatCompletion.choices || chatCompletion.choices.length === 0) {
        throw new Error('GROK Agent: No choices returned from GROK API');
      }

      const modelStats = [];
      const obj = {};
      obj["id"] = chatCompletion.id;
      obj["model"] = chatCompletion.model;
      obj["usage"] = chatCompletion.usage;
      obj["from"] = "agent-grok";
      modelStats.push(obj);
      Logger.logger.info(`GROK Agent Usage: `, JSON.stringify(obj));
      const assistantMessage = chatCompletion.choices[0]?.message?.content;

      if (typeof assistantMessage !== 'string') {
        throw new Error('GROK Agent: Unexpected response format from GROK API');
      }

      return {
        role: ParticipantRole.ASSISTANT,
        content: [{ text: assistantMessage }],
        modelStats: modelStats
      };
    } catch (error) {
      Logger.logger.error('GROK Agent: Error in GROK API call:', error);
      throw error;
    }
  }

  private async *handleStreamingResponse(options: OpenAI.Chat.ChatCompletionCreateParams): AsyncIterable<string> {
    const stream = await this.client.chat.completions.create({ ...options, stream: true });
    for await (const chunk of stream) {
      const content = chunk.choices[0]?.delta?.content;
      if (content) {
        yield content;
      }
    }
  }




}