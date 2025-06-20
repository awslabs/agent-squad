import { QdrantClient } from "@qdrant/js-client-rest";
import { ConversationMessage } from "../types";
import { ChatStorage } from "./chatStorage";
import { Logger } from "../utils/logger";


export interface QdrantConversationMessage {
    id: string;
    userId: string;
    channelId?: string;
    threadTs?: string;
    content: string;
    timestamp: number;
    messageType: 'user' | 'bot';
    metadata?: Record<string, any>;
  }

export class QdrantStorage extends ChatStorage {

    private client: QdrantClient;
    private collectionName: string;

    constructor(collectionName: string) {
        super();
        this.client = new QdrantClient({
          url: process.env.QDRANT_URL!,
          apiKey: process.env.QDRANT_API_KEY,
        });
        this.collectionName = collectionName || process.env.QDRANT_COLLECTION;
    }


    saveChatMessage(userId: string, sessionId: string, agentId: string, newMessage: ConversationMessage, maxHistorySize?: number): Promise<ConversationMessage[]> {
        let channelId = sessionId;
        let threadTs = sessionId;
        if(sessionId.includes("#")){
           const messageDetails = sessionId.split("#");
           channelId = messageDetails[0].trim();
           threadTs = messageDetails[1].trim();
        }
        const message: QdrantConversationMessage = {
            id: `${sessionId}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            userId,
            channelId,
            threadTs,
            content: newMessage.content,
            timestamp: Date.now(),
            messageType: newMessage.role === 'user' ? 'user' : 'bot',
            metadata: {
              agentId,
              sessionId,
              ...newMessage.metadata,
            },
          }
    }
    fetchChat(userId: string, sessionId: string, agentId: string, maxHistorySize?: number): Promise<ConversationMessage[]> {
        throw new Error("Method not implemented.");
    }
    fetchAllChats(userId: string, sessionId: string): Promise<ConversationMessage[]> {
        throw new Error("Method not implemented.");
    }

}