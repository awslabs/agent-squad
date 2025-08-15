// import { AgentOverlapAnalyzer } from "./agentOverlapAnalyzer";
import { Agent, AgentProcessingResult, AgentResponse } from "./agents/agent";
import { ClassifierResult } from './classifiers/classifier';
import { ChatStorage } from "./storage/chatStorage";
import { InMemoryChatStorage } from "./storage/memoryChatStorage";
import { AccumulatorTransform } from "./utils/helpers";
import { saveConversationExchange } from "./utils/chatUtils";
import { Logger } from "./utils/logger";
import { BedrockClassifier } from "./classifiers/bedrockClassifier";
import { Classifier } from "./classifiers/classifier";
import { ChatHistory, ConversationMessage, ParticipantRole } from "./types";

export interface OrchestratorConfig {
  /** If true, logs the chat interactions with the agent */
  LOG_AGENT_CHAT?: boolean;

  /** If true, logs the chat interactions with the classifier */
  LOG_CLASSIFIER_CHAT?: boolean;

  /** If true, logs the raw, unprocessed output from the classifier */
  LOG_CLASSIFIER_RAW_OUTPUT?: boolean;

  /** If true, logs the processed output from the classifier */
  LOG_CLASSIFIER_OUTPUT?: boolean;

  /** If true, logs the execution times of various operations */
  LOG_EXECUTION_TIMES?: boolean;

  /** The maximum number of retry attempts for the classifier if it receives a bad XML response */
  MAX_RETRIES?: number;

  /**
   * If true, uses the default agent when no agent is identified during intent classification.
   *
   * When set to true:
   * - If no agent is identified, the system will fall back to using a predefined default agent.
   * - This ensures that user requests are still processed, even if a specific agent cannot be determined.
   *
   * When set to false:
   * - If no agent is identified, the system will return an error message to the user.
   * - This prompts the user to rephrase their request for better agent identification.
   *
   * Use this option to balance between always providing a response (potentially less accurate)
   * and ensuring high confidence in agent selection before proceeding.
   */
  USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED?: boolean;

  /**
   * The error message to display when a classification error occurs.
   *
   * This message is shown to the user when there's an internal error during the intent classification process,
   * separate from cases where no agent is identified.
   */
  CLASSIFICATION_ERROR_MESSAGE?: string;

  /**
   * The message to display when no agent is selected to handle the user's request.
   *
   * This message is shown when the classifier couldn't determine an appropriate agent
   * and USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED is set to false.
   */
  NO_SELECTED_AGENT_MESSAGE?: string;

  /**
   * The general error message to display when an error occurs during request routing.
   *
   * This message is shown when an unexpected error occurs during the processing of a user's request,
   * such as errors in agent dispatch or processing.
   */
  GENERAL_ROUTING_ERROR_MSG_MESSAGE?: string;

  /**
   * Maximum number of message pairs (user-assistant interactions) to retain per agent.
   *
   * This constant defines the upper limit for the conversation history stored for each agent.
   * Each pair consists of a user message and its corresponding assistant response.
   *
   * Usage:
   * - When saving messages: pass (MAX_MESSAGE_PAIRS_PER_AGENT * 2) as maxHistorySize
   * - When fetching chats: pass (MAX_MESSAGE_PAIRS_PER_AGENT * 2) as maxHistorySize
   *
   * Note: The actual number of messages stored will be twice this value,
   * as each pair consists of two messages (user and assistant).
   *
   * Example:
   * If MAX_MESSAGE_PAIRS_PER_AGENT is 5, up to 10 messages (5 pairs) will be stored per agent.
   */
  MAX_MESSAGE_PAIRS_PER_AGENT?: number;
}

export const DEFAULT_CONFIG: OrchestratorConfig = {
  /** Default: Do not log agent chat interactions */
  LOG_AGENT_CHAT: false,

  /** Default: Do not log classifier chat interactions */
  LOG_CLASSIFIER_CHAT: false,

  /** Default: Do not log raw classifier output */
  LOG_CLASSIFIER_RAW_OUTPUT: false,

  /** Default: Do not log processed classifier output */
  LOG_CLASSIFIER_OUTPUT: false,

  /** Default: Do not log execution times */
  LOG_EXECUTION_TIMES: false,

  /** Default: Retry classifier up to 3 times on bad XML response */
  MAX_RETRIES: 3,

  /** Default: Use the default agent when no agent is identified during intent classification */
  USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED: true,

  /** Default error message for classification errors */
  CLASSIFICATION_ERROR_MESSAGE: undefined,

  /** Default message when no agent is selected to handle the request */
  NO_SELECTED_AGENT_MESSAGE: "I'm sorry, I couldn't determine how to handle your request. Could you please rephrase it?",

  /** Default general error message for routing errors */
  GENERAL_ROUTING_ERROR_MSG_MESSAGE: undefined,

  /** Default: Maximum of 100 message pairs (200 individual messages) to retain per agent */
  MAX_MESSAGE_PAIRS_PER_AGENT: 100,
};

export interface DispatchToAgentsParams {
  // The original input provided by the user
  userInput: string;

  // Unique identifier for the user who initiated the request
  userId: string;

  // Unique identifier for the current session
  sessionId: string;

  // The result from a classifier, determining which agent to use
  classifierResult: ClassifierResult;

  // Optional: Additional parameters or metadata to be passed to the agents
  // Can store any key-value pairs of varying types
  additionalParams?: Record<string, any>;

  chatHistory?: ChatHistory
}

/**
 * Configuration options for the Orchestrator.
 * @property storage - Optional ChatStorage instance for persisting conversations.
 * @property config - Optional partial configuration for the Orchestrator.
 * @property logger - Optional logging mechanism.
 */
export interface OrchestratorOptions {
  storage?: ChatStorage;
  config?: Partial<OrchestratorConfig>;
  logger?: any;
  classifier?: Classifier;
  defaultAgent?: Agent;
}

export interface RequestMetadata {
  // The original input provided by the user
  userInput: string;

  // Unique identifier for the agent that processed the request
  agentId: string;

  // Human-readable name of the agent
  agentName: string;

  // Unique identifier for the user who initiated the request
  userId: string;

  // Unique identifier for the current session
  sessionId: string;

  // Additional parameters or metadata related to the request
  // Stores string key-value pairs
  additionalParams: Record<string, string>;

  // Optional: Indicates if classification failed during processing
  // Only present if an error occurred during classification
  errorType?: 'classification_failed';
}


export class MultiAgentOrchestrator {
  private config: OrchestratorConfig;
  private storage: ChatStorage;
  private agents: { [key: string]: Agent };
  public classifier: Classifier;
  private executionTimes: Map<string, number>;
  private logger: Logger;
  private defaultAgent: Agent;

  constructor(options: OrchestratorOptions = {}) {
    this.storage = options.storage || new InMemoryChatStorage();
    // Merge the provided config with the DEFAULT_CONFIG
    this.config = {
      LOG_AGENT_CHAT:
        options.config?.LOG_AGENT_CHAT ?? DEFAULT_CONFIG.LOG_AGENT_CHAT,
      LOG_CLASSIFIER_CHAT:
        options.config?.LOG_CLASSIFIER_CHAT ??
        DEFAULT_CONFIG.LOG_CLASSIFIER_CHAT,
      LOG_CLASSIFIER_RAW_OUTPUT:
        options.config?.LOG_CLASSIFIER_RAW_OUTPUT ??
        DEFAULT_CONFIG.LOG_CLASSIFIER_RAW_OUTPUT,
      LOG_CLASSIFIER_OUTPUT:
        options.config?.LOG_CLASSIFIER_OUTPUT ??
        DEFAULT_CONFIG.LOG_CLASSIFIER_OUTPUT,
      LOG_EXECUTION_TIMES:
        options.config?.LOG_EXECUTION_TIMES ??
        DEFAULT_CONFIG.LOG_EXECUTION_TIMES,
      MAX_RETRIES: options.config?.MAX_RETRIES ?? DEFAULT_CONFIG.MAX_RETRIES,
      MAX_MESSAGE_PAIRS_PER_AGENT:
        options.config?.MAX_MESSAGE_PAIRS_PER_AGENT ??
        DEFAULT_CONFIG.MAX_MESSAGE_PAIRS_PER_AGENT,
      USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED:
        options.config?.USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED ??
        DEFAULT_CONFIG.USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED,
      CLASSIFICATION_ERROR_MESSAGE: options.config?.CLASSIFICATION_ERROR_MESSAGE,
      NO_SELECTED_AGENT_MESSAGE:
        options.config?.NO_SELECTED_AGENT_MESSAGE ??
        DEFAULT_CONFIG.NO_SELECTED_AGENT_MESSAGE,
      GENERAL_ROUTING_ERROR_MSG_MESSAGE: options.config?.GENERAL_ROUTING_ERROR_MSG_MESSAGE
    };

    this.executionTimes = new Map();

    this.logger = new Logger(options.config, options.logger);

    this.agents = {};
    this.classifier = options.classifier || new BedrockClassifier();

    this.defaultAgent = options.defaultAgent;

  }

  analyzeAgentOverlap(): void {
    // const agents = this.getAllAgents();
    // const analyzer = new AgentOverlapAnalyzer(agents);
    // analyzer.analyzeOverlap();
  }

  addAgent(agent: Agent): void {
    if (this.agents[agent.id]) {
      throw new Error(`An agent with ID '${agent.id}' already exists.`);
    }
    this.agents[agent.id] = agent;
    this.classifier.setAgents(this.agents);
  }

  getDefaultAgent(): Agent {
    return this.defaultAgent;
  }

  setDefaultAgent(agent: Agent): void {
    this.defaultAgent = agent;
  }

  getAllAgents(): { [key: string]: { name: string; description: string } } {
    return Object.fromEntries(
      Object.entries(this.agents).map(([key, { name, description }]) => [
        key,
        { name, description },
      ])
    );
  }

  private isAsyncIterable(obj: any): obj is AsyncIterable<any> {
    return obj != null && typeof obj[Symbol.asyncIterator] === "function";
  }

  async dispatchToAgent(
    params: DispatchToAgentsParams
  ): Promise<string | AsyncIterable<any>> {
    const {
      userInput,
      userId,
      sessionId,
      classifierResult,
      additionalParams = {},
      chatHistory
    } = params;

    try {
      if (!classifierResult.selectedAgent) {
        return "I'm sorry, but I need more information to understand your request. Could you please be more specific?";
      } else {
        const { selectedAgent } = classifierResult;
        // const agentChatHistory = await this.storage.fetchChat(
        //   userId,
        //   sessionId,
        //   selectedAgent.id
        // );

        // this.logger.printChatHistory(agentChatHistory, selectedAgent.id);

        this.logger.info(
          `Routing intent "${userInput}" to ${selectedAgent.id} ...`
        );

        // const response = await this.measureExecutionTime(
        //   `Agent ${selectedAgent.name} | Processing request`,
        //   async () =>
        //     await selectedAgent.processRequest(
        //       userInput,
        //       userId,
        //       sessionId,
        //       agentChatHistory,
        //       additionalParams
        //     )
        // );

        const response =  await selectedAgent.processRequest(
          userInput,
          userId,
          sessionId,
          chatHistory,
          additionalParams
        )

        //if (this.isStream(response)) {
        if (this.isAsyncIterable(response)) {
          return response;
        }

        classifierResult.modelStats.push(...response.modelStats);
        classifierResult.info =  {};
        if(response.citations){
          classifierResult.info["citations"] = response.citations;
        }

        let responseText = "No response content";
        if (
          response.content &&
          response.content.length > 0 &&
          response.content[0].text
        ) {
          responseText = response.content[0].text;
        }else{
          this.logger.info(
            `No Response content: ${JSON.stringify(response)}`
          );
        }

        return responseText;
      }
    } catch (error) {
      this.logger.error("Error during agent dispatch:", error);
      throw error;
    }
  }

  async classifyRequest(
    userInput: string,
    userId: string,
    sessionId: string, 
    chatHistory: ChatHistory
  ): Promise<ClassifierResult[]> {
    try {
      // const chatHistory = await this.storage.fetchAllChats(userId, sessionId) || [];
      // const classifierResult = await this.measureExecutionTime(
      //   "Classifying user intent",
      //   () => this.classifier.classify(userInput, chatHistory)
      // );

      this.logger.info("Classifying user intent");
      let classifierResults;
      try{
        classifierResults = await this.classifier.classify(userInput, chatHistory);
      }catch(e){
        this.logger.error("There was an error during classification: ",e);
        this.logger.info("Default agent to be used due to an error");
        classifierResults=[];
        const classifierResult=this.getFallbackResult(null);
        classifierResult["modelStats"]["from"] = "classifier_error";
        classifierResult["modelStats"]["usage"] = {};
        classifierResults.push(classifierResult);
      }
      
      for(const [index,classifierResult] of classifierResults.entries()){
        this.logger.printIntent(userInput, classifierResult);
        if (!classifierResult.selectedAgent && this.config.USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED && this.defaultAgent) {
          this.logger.info("Using default agent as no agent was selected at index: ", index);
          classifierResults[index]  = this.getFallbackResult(classifierResult.modelStats);;
        }
      }
  
      return classifierResults;
    } catch (error) {
      this.logger.error("Error during intent classification:", error);
      throw error;
    }
  }
  
  async agentProcessRequest(
    userInput: string,
    userId: string,
    sessionId: string,
    classifierResult: ClassifierResult,
    additionalParams: Record<any, any> = {},
    chatHistory: ChatHistory
  ): Promise<AgentResponse> {
    try {
      const agentResponse = await this.dispatchToAgent({
        userInput,
        userId,
        sessionId,
        classifierResult,
        additionalParams,
        chatHistory
      });
  
      const metadata = this.createMetadata(classifierResult, userInput, userId, sessionId, additionalParams);
  
      if (this.isAsyncIterable(agentResponse)) {
        const accumulatorTransform = new AccumulatorTransform();
        this.processStreamInBackground(
          agentResponse,
          accumulatorTransform,
          userInput,
          userId,
          sessionId,
          classifierResult.selectedAgent
        );
        return {
          metadata,
          output: accumulatorTransform,
          streaming: true,
          modelStats:  [{"a":"streaming"}]
        };
      }
  
      if (classifierResult?.selectedAgent.saveChat) {
        await saveConversationExchange(
          userInput,
          agentResponse,
          this.storage,
          userId,
          sessionId,
          classifierResult?.selectedAgent.id,
          this.config.MAX_MESSAGE_PAIRS_PER_AGENT
        );
      }
  
      return {
        metadata,
        output: agentResponse,
        streaming: false,
        modelStats: classifierResult.modelStats,
        info: classifierResult.info
      };
    } catch (error) {
      this.logger.error("Error during agent processing:", error);
      throw error;
    }
  }
  
  async routeRequest(
    userInput: string,
    userId: string,
    sessionId: string,
    additionalParams: Record<any, any> = {}
  ): Promise<AgentResponse> {
    this.executionTimes = new Map();
  
    // let modelStats = [];
    try {
      const chatHistory = await this.storage.fetchAllChats(userId, sessionId, userInput) ||  { messages : []} ;
      this.logger.printChatHistory(chatHistory.messages);
      this.logger.info(`Chat Summary: ${chatHistory.summary}`);
      const classifierResults = await this.classifyRequest(userInput, userId, sessionId, chatHistory);

      //iterate over classified results
      let agentResponse;
      const combinedMetadata = {} as Omit<AgentProcessingResult, 'response'>;
      const info = [];
      for(const classifierResult of classifierResults){
        agentResponse =  await this.agentProcessRequest(userInput, userId, sessionId, classifierResult, additionalParams, chatHistory);

        const updatedMessage = {} as ConversationMessage;
        updatedMessage["role"] = ParticipantRole.ASSISTANT;
        updatedMessage["content"] = [{"text":agentResponse.output}];
        chatHistory.messages.push(updatedMessage);

        this.combineAgentResponses(combinedMetadata, info, classifierResult, agentResponse);
      }

      //update data for all agents invoked.
      //modelstats is not needed to be combined as its the same array being set inside classifier.
      agentResponse.info =info;
      agentResponse.metadata = combinedMetadata;
      return agentResponse;

    } catch (error) {
      console.log(error);
      return {
        metadata: this.createMetadata(null, userInput, userId, sessionId, additionalParams),
        output: this.config.GENERAL_ROUTING_ERROR_MSG_MESSAGE || String(error),
        streaming: false,
        modelStats: []
      };
    } finally {
      this.logger.printExecutionTimes(this.executionTimes);
    }
  }

  /**
   * Combined responses from multiple agent calls and classifier into one.
   * We dont need to combine modelstats as the same array is set against all classifier entries. so
   * it gets updated automatically within all agents that is called.
   * @param combinedMetadata 
   * @param info 
   * @param classifierResult 
   * @param agentResponse 
   * @returns 
   */
  private combineAgentResponses(combinedMetadata: Omit<AgentProcessingResult, 'response'>,info: any, classifierResult: ClassifierResult, agentResponse: AgentResponse) {
    this.combineMetadata(combinedMetadata, agentResponse.metadata);
    const agentInfo = agentResponse.info;
    if(!agentInfo?.citations){
      return
    }
    if(!info["citations"]){
      info["citations"] = [];
    }

    info["citations"].push(...agentInfo["citations"]);
  }

  private combineMetadata(combinedMetadata, metadata){
    if(!combinedMetadata.agentId){
      Object.assign(combinedMetadata, metadata);
    }else{
      combinedMetadata.agentId = combinedMetadata.agentId+","+metadata.agentId;
      combinedMetadata.agentName = combinedMetadata.agentName+","+metadata.agentName;
    }
  }
  
  private async processStreamInBackground(
    agentResponse: AsyncIterable<any>,
    accumulatorTransform: AccumulatorTransform,
    userInput: string,
    userId: string,
    sessionId: string,
    agent: Agent
  ): Promise<void> {
    const streamStartTime = Date.now();
    let chunkCount = 0;

    try {
      for await (const chunk of agentResponse) {
        if (chunkCount === 0) {
          const firstChunkTime = Date.now();
          const timeToFirstChunk = firstChunkTime - streamStartTime;
          this.executionTimes.set("Time to first chunk", timeToFirstChunk);
          this.logger.printExecutionTimes(this.executionTimes);
        }
        accumulatorTransform.write(chunk);
        chunkCount++;
      }

      accumulatorTransform.end();
      this.logger.debug(`Streaming completed: ${chunkCount} chunks received`);

      const fullResponse = accumulatorTransform.getAccumulatedData();
      if (fullResponse) {



      if (agent.saveChat) {
        await saveConversationExchange(
          userInput,
          fullResponse,
          this.storage,
          userId,
          sessionId,
          agent.id
        );
      }

      } else {
        this.logger.warn("No data accumulated, messages not saved");
      }
    } catch (error) {
      this.logger.error("Error processing stream:", error);
      accumulatorTransform.end();
      if (error instanceof Error) {
        accumulatorTransform.destroy(error);
      } else if (typeof error === "string") {
        accumulatorTransform.destroy(new Error(error));
      } else {
        accumulatorTransform.destroy(new Error("An unknown error occurred"));
      }
    }
  }

  private measureExecutionTime<T>(
    timerName: string,
    fn: () => Promise<T> | T
  ): Promise<T> {
    if (!this.config.LOG_EXECUTION_TIMES) {
      return Promise.resolve(fn());
    }

    const startTime = Date.now();
    this.executionTimes.set(timerName, startTime);

    return Promise.resolve(fn()).then(
      (result) => {
        const endTime = Date.now();
        const duration = endTime - startTime;
        this.executionTimes.set(timerName, duration);
        return result;
      },
      (error) => {
        const endTime = Date.now();
        const duration = endTime - startTime;
        this.executionTimes.set(timerName, duration);
        throw error;
      }
    );
  }

  private createMetadata(
    intentClassifierResult: ClassifierResult | null,
    userInput: string,
    userId: string,
    sessionId: string,
    additionalParams: Record<string, string>
  ): RequestMetadata {
    const baseMetadata = {
      userInput,
      userId,
      sessionId,
      additionalParams,
    };

    if (!intentClassifierResult || !intentClassifierResult.selectedAgent) {
      return {
        ...baseMetadata,
        agentId: "no_agent_selected",
        agentName: "No Agent",
        errorType: "classification_failed",
      };
    }

    return {
      ...baseMetadata,
      agentId: intentClassifierResult.selectedAgent.id,
      agentName: intentClassifierResult.selectedAgent.name,
    };
  }

  private getFallbackResult(m:[] | null | undefined ): ClassifierResult {
    return {
      selectedAgent: this.getDefaultAgent(),
      confidence: 0,
      modelStats: m ?? []
     };
  }

}
