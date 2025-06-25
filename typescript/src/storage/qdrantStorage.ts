import { QdrantClient } from "@qdrant/js-client-rest";
import { ChatHistory, ConversationMessage, ParticipantRole } from "../types";
import { ChatStorage } from "./chatStorage";
import { QdrantUtils } from "../utils/qdrantUtils";
import { Logger } from "../utils/logger";
import { LLMUtils } from "../utils/llmUtils";
import { v4 as uuidv4 } from "uuid";

export interface QdrantConversationMessage {
  id: string;
  userId: string;
  channelId?: string;
  threadTs?: string;
  content: string;
  timestamp: number;
  messageType: "user" | "bot";
  metadata?: Record<string, any>;
  isDM: boolean;
}

export interface QdrantPoint {
  id: string;
  vector: number[];
  payload: {
    contextId: string;
    contextType: "channel_thread" | "user_dm";
    userId: string;
    channelId?: string;
    threadTs?: string;
    content: string;
    messageType: "user" | "bot";
    timestamp: number;
    messageIndex: number;
    isActive: boolean;
    metadata?: Record<string, any>;
  };
}

export class QdrantStorage extends ChatStorage {
  private client: QdrantClient;
  private collectionName: string;
  private qdrantUtils: QdrantUtils;
  private llmUtils: LLMUtils;
  private maxCount: number;

  constructor(collectionName: string, llmUtils: LLMUtils, maxCount: number) {
    super();
    this.client = new QdrantClient({
      url: process.env.QDRANT_URL!,
      apiKey: process.env.QDRANT_API_KEY,
    });
    this.collectionName = collectionName || process.env.QDRANT_COLLECTION;
    this.qdrantUtils = new QdrantUtils();
    this.llmUtils = llmUtils;
    this.maxCount = maxCount || 20;
  }

  async saveChatMessage(
    userId: string,
    sessionId: string,
    agentId: string,
    newMessage: ConversationMessage
  ): Promise<ConversationMessage[]> {
    const [channelId, threadTs] = this.parseSessionId(sessionId);
    const message: QdrantConversationMessage = {
      id: '',
      userId,
      channelId,
      threadTs,
      content: newMessage.content[0].text,
      timestamp: Date.now(),
      messageType: newMessage.role === "user" ? "user" : "bot",
      metadata: {
        agentId,
        sessionId,
        channelId,
        threadTs,
      },
      isDM: userId === sessionId,
    };

    await this.storeMessage(message);
    const retMessages = [];
    retMessages.push(newMessage);
    return retMessages;
  }
  async fetchChat(userId: string, sessionId: string): Promise<ConversationMessage[]> {
    const [channelId, threadTs] = this.parseSessionId(sessionId);
    const point = await this.getPoint(userId, channelId, threadTs);
    const payload = point.payload
      const conversations = [];
      conversations.push({
        role: payload.messageType === 'user' ? ParticipantRole.USER : ParticipantRole.ASSISTANT,
        content: payload.content,
        timestamp: payload.timestamp,
        metadata: payload.metadata,
      });
      return conversations;
  }

  async fetchAllChats(
    userId: string,
    sessionId: string,
    query: string
  ): Promise<ChatHistory> {

    const [channelId, threadTs] = this.parseSessionId(sessionId);
    const context = await this.getRelevantContext(
        query, 
        userId,
        channelId,
        threadTs,
        this.maxCount
      );

      const messages = context.messages.map(msg => ({
        role: msg.payload.messageType === 'user' ? ParticipantRole.USER : ParticipantRole.ASSISTANT,
        content: [{text: msg.payload.content}],
        timestamp: msg.payload.timestamp,
        metadata: msg.payload.metadata,
      }));

      return { messages, summary: context.summary };

  }

  async getRelevantContext(
    query: string,
    userId: string,
    channelId?: string,
    threadTs?: string,
    maxResults: number = 10
  ): Promise<{ messages: any[]; summary?: string }> {
    try {
      // Generate context ID for the query
      const contextId = this.generateContextId({
        id: '',
        userId,
        channelId,
        threadTs,
        content: query,
        timestamp: Date.now(),
        messageType: 'user',
      } as QdrantConversationMessage);

      let recentMessages: any[] = [];

      // Generate query embedding
      const queryEmbedding = await this.qdrantUtils.generateEmbedding(query);

      // Search for similar messages
      const similarMessages = await this.searchSimilar(
        queryEmbedding,
        contextId,
        Math.floor(maxResults * 0.7), // 70% from similarity search
        0.7
      );

      // Get remaining messages from recent
      const remainingCount = maxResults - similarMessages.length;
      if (remainingCount > 0) {
        recentMessages = await this.getRecentMessages(
          contextId,
          remainingCount
        );
      }

      // Combine and deduplicate
      const allMessages = [...similarMessages, ...recentMessages];
      const uniqueMessages = Array.from(
        new Map(allMessages.map(msg => [msg.id, msg])).values()
      );

      // Sort by timestamp
      uniqueMessages.sort((a, b) => 
        (a.payload?.timestamp || 0) - (b.payload?.timestamp || 0)
      );

      // Get context summary if available
      const summary = await this.getContextSummary(contextId);

      return {
        messages: uniqueMessages.slice(-maxResults),
        summary,
      };
    } catch (error) {
      console.error('Error getting relevant context:', error);
      throw error;
    }
  }

  async getPoint(userId, channelId, threadTs){
    const id = this.formContextId(userId, channelId, threadTs);
    const point = await this.client.query(this.collectionName, {
        query: id
      })
      return point[0];
  }

  async searchSimilar(
    queryVector: number[],
    contextId: string,
    limit: number = 10,
    scoreThreshold: number = 0.7
  ): Promise<any[]> {
    try {
      const searchResult = await this.client.search(this.collectionName, {
        vector: queryVector,
        filter: {
          must: [
            { key: 'contextId', match: { value: contextId } },
            { key: 'isActive', match: { value: true } }
          ],
        },
        limit,
        score_threshold: scoreThreshold,
        with_payload: true,
      });

      return searchResult;
    } catch (error) {
      console.error('Error searching Qdrant:', error);
      throw error;
    }
  }

  async storeMessage(message: QdrantConversationMessage): Promise<void> {
    try {
      const contextId = this.generateContextId(message);
      const contextType = this.getContextType(message);

      // Generate embedding for the message
      const embedding = await this.qdrantUtils.generateEmbedding(
        message.content
      );

      // Get current message count for this context
      const recentMessages = await this.getRecentMessages(contextId, 1000);
      const messageIndex = recentMessages.length;

      // const pointId = await this.generatePointId(message, messageIndex);
      const pointId = uuidv4();

      // Create Qdrant point
      const point: QdrantPoint = {
        id: pointId,
        vector: embedding,
        payload: {
          contextId,
          contextType,
          userId: message.userId,
          channelId: message.channelId,
          threadTs: message.threadTs,
          content: message.content,
          messageType: message.messageType,
          timestamp: message.timestamp,
          messageIndex,
          isActive: true,
          metadata: message.metadata,
        },
      };

      // Store the point
      await this.storePoints([point]);

      // Context pruning - keep only recent messages active
      const maxActiveMessages = contextType === "channel_thread" ? 100 : 200;
      await this.deactivateOldMessages(contextId, maxActiveMessages);

      // Summarization for large contexts
      if (messageIndex > 0 && messageIndex % 50 === 0) {
        await this.createContextSummary(contextId);
      }
      Logger.logger.info("Saved to vectorstore");
    } catch (error) {
      Logger.logger.error("Error storing message:", error);
      throw error;
    }
  }

  private generatePointId(message: QdrantConversationMessage, messageIndex: number): string {
    const contextId = this.generateContextId(message);
    
    if(message.isDM){
      return `dm_${message.userId}_msg_${messageIndex.toString().padStart(6, '0')}`;
    }else{
      return `${contextId}_msg_${messageIndex.toString().padStart(6, '0')}`;
    }
  }

  private generateContextId(message: QdrantConversationMessage): string {
    if (message.isDM) {
      // DM context
      return `dm_${message.userId}`;
    } else {
      return `channel_${message.channelId}_thread_${message.threadTs}`;
    }
  }

  private formContextId(userId, channelId, threadTs){
    if(userId === channelId){
        return `dm_${userId}`;
    }else{
        return `channel_${channelId}_thread_${threadTs}`;
    }
  }

  private getContextType(
    message: QdrantConversationMessage
  ): "channel_thread" | "user_dm" {
    return !message.isDM ? "channel_thread" : "user_dm";
  }

  async getRecentMessages(
    contextId: string,
    limit: number = 20
  ): Promise<any[]> {
    try {
      const searchResult = await this.client.scroll(this.collectionName, {
        filter: {
          must: [
            { key: "contextId", match: { value: contextId } },
            { key: "isActive", match: { value: true } },
          ],
        },
        limit,
        with_payload: true,
        order_by: { key: "timestamp", direction: "desc" },
      });

      return searchResult.points || [];
    } catch (error) {
      console.error("Error getting recent messages:", error);
      throw error;
    }
  }

  async storePoints(points: QdrantPoint[]): Promise<void> {
    try {
      await this.client.upsert(this.collectionName, {
        wait: true,
        points: points.map((point) => ({
          id: point.id,
          vector: point.vector,
          payload: point.payload,
        })),
      });
    } catch (error) {
      console.error("Error storing points in Qdrant:", error);
      throw error;
    }
  }

  async deactivateOldMessages(
    contextId: string,
    keepRecent: number = 50
  ): Promise<void> {
    try {
      // Get all messages for context, sorted by timestamp desc
      const allMessages = await this.client.scroll(this.collectionName, {
        filter: {
          must: [
            { key: "contextId", match: { value: contextId } },
            { key: "isActive", match: { value: true } },
          ],
        },
        limit: 1000,
        with_payload: true,
        order_by: { key: "timestamp", direction: "desc" },
      });

      const messages = allMessages.points || [];
      if (messages.length <= keepRecent) return;

      // Deactivate old messages
      const toDeactivate = messages.slice(keepRecent);
      const updatePoints = toDeactivate.map((point) => ({
        id: point.id,
        payload: { ...point.payload, isActive: false },
      }));

      await this.client.setPayload(this.collectionName, {
        points: updatePoints.map((p) => p.id),
        payload: { isActive: false },
        wait: true,
      });
    } catch (error) {
      console.error("Error deactivating old messages:", error);
      throw error;
    }
  }

  private async createContextSummary(contextId: string): Promise<void> {
    try {
      // Get recent messages for summarization
      const messages = await this.getRecentMessages(contextId, 50);

      if (messages.length < 10) return; // Not enough messages to summarize

      const conversationText = messages
        .sort(
          (a, b) => (a.payload?.timestamp || 0) - (b.payload?.timestamp || 0)
        )
        .map((msg) => `${msg.payload?.messageType}: ${msg.payload?.content}`)
        .join("\n");

      const summary = await this.llmUtils.generateSummary(conversationText);

      // Store summary as a special message
      const summaryEmbedding =
        await this.qdrantUtils.generateEmbedding(summary);
      const summaryPoint: QdrantPoint = {
        id: uuidv4(),
        vector: summaryEmbedding,
        payload: {
          contextId,
          contextType: contextId.startsWith("channel_")
            ? "channel_thread"
            : "user_dm",
          userId: "system",
          content: summary,
          messageType: "bot",
          timestamp: Date.now(),
          messageIndex: -1,
          isActive: true,
          metadata: { type: "summary" },
        },
      };

      await this.storePoints([summaryPoint]);
    } catch (error) {
      console.error("Error creating context summary:", error);
    }
  }

  private parseSessionId(
    sessionId: string
  ): [string | undefined, string | undefined] {
    const parts = sessionId.split("#");
    if (parts.length == 2) {
      return [parts[0], parts[1]];
    }
    return [sessionId, sessionId];
  }

  private async getContextSummary(contextId: string): Promise<string | undefined> {
    try {
      const summaryMessages = await this.client.scroll(
        this.collectionName,
        {
          filter: {
            must: [
              { key: 'contextId', match: { value: contextId } },
              { key: 'metadata.type', match: { value: 'summary' } },
              { key: 'isActive', match: { value: true } }
            ],
          },
          limit: 1,
          with_payload: true,
          order_by: { key: 'timestamp', direction: 'desc' },
        }
      );

      const latestSummary = summaryMessages.points?.[0];
      return latestSummary?.payload?.content as string;
    } catch (error) {
      console.error('Error getting context summary:', error);
      return undefined;
    }
  }
}
