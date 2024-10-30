import {
  BedrockRuntimeClient,
  ConverseCommand,
  ConverseStreamCommand,
  Tool,
} from "@aws-sdk/client-bedrock-runtime";
import { Agent, AgentOptions } from "./agent";
import {
  BEDROCK_MODEL_ID_CLAUDE_3_HAIKU,
  ConversationMessage,
  ParticipantRole,
  TemplateVariables,
} from "../types";
import { Retriever } from "../retrievers/retriever";
import { Logger } from "../utils/logger"

export interface BedrockLLMAgentOptions extends AgentOptions {
  streaming?: boolean;
  inferenceConfig?: {
    maxTokens?: number;
    temperature?: number;
    topP?: number;
    stopSequences?: string[];
  };
  guardrailConfig?: {
    guardrailIdentifier: string;
    guardrailVersion: string;
  };
  retriever?: Retriever;
  toolConfig?: {
    tool: Tool[];
    useToolHandler: (response: any, conversation: ConversationMessage[]) => any ;
    toolMaxRecursions?: number;
  };
  customSystemPrompt?: {
    template: string, variables?: TemplateVariables
  };
  client?: BedrockRuntimeClient;
}

/**
 * BedrockAgent class represents an agent that uses Amazon Bedrock for natural language processing.
 * It extends the base Agent class and implements the processRequest method using Bedrock's API.
 */
export class BedrockLLMAgent extends Agent {
  /** AWS Bedrock Runtime Client for making API calls */
  protected client: BedrockRuntimeClient;

  protected customSystemPrompt?: string;

  protected streaming: boolean;

  protected inferenceConfig: {
    maxTokens?: number;
    temperature?: number;
    topP?: number;
    stopSequences?: string[];
  };

  /**
   * The ID of the model used by this agent.
   */
  protected modelId?: string;

  protected guardrailConfig?: {
    guardrailIdentifier: string;
    guardrailVersion: string;
  };

  protected retriever?: Retriever;

  private toolConfig?: {
    tool: any[];
    useToolHandler: (response: any, conversation: ConversationMessage[]) => any;
    toolMaxRecursions?: number;
  };

  private promptTemplate: string;
  private systemPrompt: string;
  private customVariables: TemplateVariables;
  private defaultMaxRecursions: number = 20;

  /**
   * Constructs a new BedrockAgent instance.
   * @param options - Configuration options for the agent, inherited from AgentOptions.
   */
  constructor(options: BedrockLLMAgentOptions) {
    super(options);

    this.client = options.client ? options.client : options.region
      ? new BedrockRuntimeClient({ region: options.region })
      : new BedrockRuntimeClient();

    // Initialize the modelId
    this.modelId = options.modelId ?? BEDROCK_MODEL_ID_CLAUDE_3_HAIKU;

    this.streaming = options.streaming ?? false;

    this.inferenceConfig = options.inferenceConfig ?? {};

    this.guardrailConfig = options.guardrailConfig ?? null;

    this.retriever = options.retriever ?? null;

    this.toolConfig = options.toolConfig ?? null;

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

    if (options.customSystemPrompt) {
      this.setSystemPrompt(
        options.customSystemPrompt.template,
        options.customSystemPrompt.variables
      );
    }

  }

  /**
   * Abstract method to process a request.
   * This method must be implemented by all concrete agent classes.
   *
   * @param inputText - The user input as a string.
   * @param chatHistory - An array of Message objects representing the conversation history.
   * @param additionalParams - Optional additional parameters as key-value pairs.
   * @returns A Promise that resolves to a Message object containing the agent's response.
   */
  /* eslint-disable @typescript-eslint/no-unused-vars */
  async processRequest(
    inputText: string,
    userId: string,
    sessionId: string,
    chatHistory: ConversationMessage[],
    additionalParams?: Record<string, string>
  ): Promise<ConversationMessage | AsyncIterable<any>> {
    try {

      // Construct the user's message based on the provided inputText
      const userMessage: ConversationMessage = {
        role: ParticipantRole.USER,
        content: [{ text: `${inputText}` }],
      };

      // Combine the existing chat history with the user's message
      const conversation: ConversationMessage[] = [
        ...chatHistory,
        userMessage,
      ];

      this.updateSystemPrompt();

      let systemPrompt = this.systemPrompt;

      // Update the system prompt with the latest history, agent descriptions, and custom variables
      if (this.retriever) {
        // retrieve from Vector store
        const response = await this.retriever.retrieveAndCombineResults(inputText);
        const contextPrompt =
          "\nHere is the context to use to answer the user's question:\n" +
          response;
          systemPrompt = systemPrompt + contextPrompt;
      }

      // Prepare the command to converse with the Bedrock API
      const converseCmd = {
        modelId: this.modelId,
        messages: conversation, //Include the updated conversation history
        system: [{ text: systemPrompt }],
        inferenceConfig: {
          maxTokens: this.inferenceConfig.maxTokens,
          temperature: this.inferenceConfig.temperature,
          topP: this.inferenceConfig.topP,
          stopSequences: this.inferenceConfig.stopSequences,
        },
        guardrailConfig: this.guardrailConfig? this.guardrailConfig:undefined,
        toolConfig: (this.toolConfig ? { tools:this.toolConfig.tool}:undefined)
      };

      if (this.streaming){
        return this.handleStreamingResponse(converseCmd);
      } else {
          let continueWithTools = false;
          let finalMessage:ConversationMessage = { role: ParticipantRole.USER, content:[]};
          let maxRecursions = this.toolConfig?.toolMaxRecursions || this.defaultMaxRecursions;

          do{
            // send the conversation to Amazon Bedrock
            const bedrockResponse = await this.handleSingleResponse(converseCmd);

            // Append the model's response to the ongoing conversation
            conversation.push(bedrockResponse);

            // process model response
            if (bedrockResponse?.content?.some((content) => 'toolUse' in content)){
              // forward everything to the tool use handler
              if (!this.toolConfig){
                throw new Error("Tool config is not defined");
              }
              const toolResponse = await this.toolConfig.useToolHandler(bedrockResponse, conversation);
              continueWithTools = true;
              converseCmd.messages.push(toolResponse);
            }
            else {
              continueWithTools = false;
              finalMessage = bedrockResponse;
            }
            maxRecursions--;

            converseCmd.messages = conversation;

          }while (continueWithTools && maxRecursions > 0)
          return finalMessage;
      }
    } catch (error) {
      Logger.logger.error("Error processing request:", error.message);
      throw `Error processing request: ${error.message}`;
    }
  }

  protected async handleSingleResponse(input: any): Promise<ConversationMessage> {
    try {
      const command = new ConverseCommand(input);

      const response = await this.client.send(command);
      if (!response.output) {
        throw new Error("No output received from Bedrock model");
      }
      return response.output.message as ConversationMessage;
    } catch (error) {
      Logger.logger.error("Error invoking Bedrock model:", error.message);
      throw `Error invoking Bedrock model: ${error.message}`;
    }
  }

  private async *handleStreamingResponse(input: any): AsyncIterable<string> {
      let toolBlock:any = {toolUseId:'', input:{}, name:''};
      let inputString = '';
      let toolUse = false;
      let recursions = this.toolConfig?.toolMaxRecursions || this.defaultMaxRecursions;

      try {
          do
          {
              const command = new ConverseStreamCommand(input);
              const response = await this.client.send(command);
              if (!response.stream) {
                  throw new Error("No stream received from Bedrock model");
              }
              for await (const chunk of response.stream) {
                  if (chunk.contentBlockDelta && chunk.contentBlockDelta.delta && chunk.contentBlockDelta.delta.text) {
                      yield chunk.contentBlockDelta.delta.text;
                  } else if (chunk.contentBlockStart?.start?.toolUse){
                      toolBlock = chunk.contentBlockStart?.start?.toolUse
                  } else if (chunk.contentBlockDelta?.delta?.toolUse){
                      inputString += chunk.contentBlockDelta.delta.toolUse.input
                  } else if (chunk.messageStop?.stopReason === 'tool_use'){
                      toolBlock.input = JSON.parse(inputString);
                      const message = {role:ParticipantRole.ASSISTANT, content:[{'toolUse':toolBlock}]}
                      input.messages.push(message);
                      const toolResponse = await this.toolConfig!.useToolHandler(message, input.messages);
                      input.messages.push(toolResponse);
                      toolUse = true;
                  } else if (chunk.messageStop?.stopReason === 'end_turn') {
                      toolUse=false;
                  }
              }
          } while (toolUse &&  --recursions > 0)
      } catch (error) {
          Logger.logger.error("Error getting stream from Bedrock model:", error.message);
          throw `Error getting stream from Bedrock model: ${error.message}`;
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
}
