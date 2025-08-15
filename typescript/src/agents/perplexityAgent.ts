import {Agent, AgentOptions} from "./agent";
import { Logger } from '../utils/logger';
import { Retriever } from "../retrievers/retriever";
import {ConversationMessage,TemplateVariables, ParticipantRole, PERPLEXITY_MODEL_ID_SONAR_PRO, ChatHistory} from "../types"
import axios, { AxiosRequestConfig } from 'axios';

const DEFAULT_MAX_TOKENS = 4096;

export interface PerplexityAgentOptions extends AgentOptions {
    model: string;
    apiKey: string;
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

export class PerplexityAgent extends Agent {

    private model: string;
    private systemPrompt: string;
    private logRequest?: boolean;
    private apiKey: string;
    protected retriever?: Retriever;
    private promptTemplate: string;
    private customVariables: TemplateVariables;
    private inferenceConfig: {
        maxTokens?: number;
        temperature?: number;
        topP?: number;
        stopSequences?: string[];
      };

    constructor(options: PerplexityAgentOptions) {
      super(options);

      if (!options.apiKey) {
        throw new Error("Perplexity API key is required");
      }
      this.apiKey = options.apiKey;
      this.model = options.model ?? PERPLEXITY_MODEL_ID_SONAR_PRO;
      this.retriever = options.retriever ?? null;
      this.logRequest =  options.logRequest ?? false;

      this.inferenceConfig = {
        maxTokens: options.inferenceConfig?.maxTokens ?? DEFAULT_MAX_TOKENS,
        temperature: options.inferenceConfig?.temperature,
        topP: options.inferenceConfig?.topP,
        stopSequences: options.inferenceConfig?.stopSequences,
      };

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
    - Seamlessly transition between topics as the human introduces new subjects.`;

    this.customVariables = {};
    this.systemPrompt = '';

    if (options.customSystemPrompt) {
      this.setSystemPrompt(
        options.customSystemPrompt.template,
        options.customSystemPrompt.variables
      );
    }

    }//constructor


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
//additionalParams?: Record<string, string> : not being used as of now. Add if needed.
    async processRequest(
        inputText: string,
        userId: string,
        sessionId: string,
        chatHistory: ChatHistory
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
          role: msg.role.toLowerCase(),
          content: msg.content[0]?.text || ''
        })),
        { role: 'user' as const, content: inputText }
      ] ;

    //   const { maxTokens, temperature, topP, stopSequences } = this.inferenceConfig;
      const data: any= {};
      data.model = this.model;
      data.messages = messages;
      const requestOptions: AxiosRequestConfig =  {
        url: `https://api.perplexity.ai/chat/completions`,
        method: 'POST',
        data: JSON.stringify(data),
        headers: {Authorization: `Bearer ${this.apiKey}`, 'Content-Type': 'application/json'},
      }
      let perplexityResp: any;
      try{
        if(this.logRequest){
          console.log("\n\n---- Perplexity Agent ----");
          console.log(JSON.stringify(requestOptions));
        }
        const retVal = await axios(requestOptions);
        if(this.logRequest){
          console.log(JSON.stringify(retVal?.data));
          console.log("\n\n");
        }
        perplexityResp = retVal?.data;
        if (!perplexityResp || perplexityResp.choices.length<1) {
            throw new Error('Perplexity Agent: Unexpected response format from Perplexity API');
        }

        const modelStats = [];
        const obj = {};
        obj["id"] = perplexityResp.id;
        obj["model"] = perplexityResp.model;
        obj["usage"] = perplexityResp.usage;
        obj["from"] = "agent-perplexity";
        modelStats.push(obj);
        Logger.logger.info(`Perplexity Agent Usage: `, JSON.stringify(obj));
        const assistantMessage = perplexityResp.choices[0]?.message?.content;

        if (typeof assistantMessage !== 'string') {
            throw new Error('Perplexity Agent: Unexpected response format from Perplexity API');
        }
        return {
            role: ParticipantRole.ASSISTANT,
            content: [{ text: assistantMessage }],
            modelStats: modelStats,
            citations: perplexityResp.citations
        };
      }catch(e){
        if(e.response){
            Logger.logger.error('Perplexity Agent: Error in Perplexity API call:', e.response);
        }else{
            Logger.logger.error('Perplexity Agent: Error in Perplexity API call:', e);
        }
        return {
          role: ParticipantRole.ASSISTANT,
          content: [{ text: "An error occured while connecting to perplexity." }],
          modelStats: [],
          citations: []
        };
      }

      
    }//processRequest

    }//class

export default PerplexityAgent;