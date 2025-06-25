import {
    S3Client,
    GetObjectCommand,
  } from "@aws-sdk/client-s3";
  
export async function fetchDescription(
  bucket: string,
  key: string
): Promise<string> {
  if (!bucket || !key) {
    throw new Error("Invalid S3 path format. Need bucket and key");
  }

  // Create S3 client
  const s3Client = new S3Client({
    region: process.env.REGION || "us-east-1",
  });

  // Get object from S3
  const resp = await s3Client.send(
    new GetObjectCommand({
      Bucket: bucket,
      Key: key,
    })
  );

  const data = JSON.parse((await resp.Body.transformToString()) || "{}");
  const description = data.description || "";

  return description;
}
