import { promises as fs } from 'fs';
import { join } from 'path';
import { createHash } from 'crypto';
import { and, eq } from 'drizzle-orm';
import { db } from '../../db/index.js';
import {
  evolutionAssets,
  evolutionDesigns,
  evolutionRequests,
  evolutionRuns,
  evolutionScreenshots,
} from '../../db/schema.js';
import { cacheService } from '../cache.js';

const SCREENSHOT_DIR = join(process.cwd(), 'storage', 'evolution-screenshots');

const ensureScreenshotDir = async () => {
  await fs.mkdir(SCREENSHOT_DIR, { recursive: true });
};

const extensionForContentType = (contentType: string) => {
  if (contentType.includes('png')) return '.png';
  if (contentType.includes('jpeg') || contentType.includes('jpg')) return '.jpg';
  if (contentType.includes('webp')) return '.webp';
  return '';
};

const hashBuffer = (buffer: Buffer) =>
  createHash('sha256').update(buffer).digest('hex');

const getOrCreateRunId = async (requestId: number, userId: string) => {
  const evolutionId = `request-${requestId}`;
  const existing = await db.query.evolutionRuns.findFirst({
    where: and(
      eq(evolutionRuns.userId, userId),
      eq(evolutionRuns.evolutionId, evolutionId)
    ),
  });

  if (existing) {
    return existing.id;
  }

  const [inserted] = await db
    .insert(evolutionRuns)
    .values({
      userId,
      evolutionId,
      name: `Request ${requestId}`,
      status: 'running',
    })
    .returning();

  return inserted.id;
};

export type UploadScreenshotInput = {
  userId: string;
  designId: number;
  dataBase64: string;
  contentType: string;
  kind?: string;
  width?: number;
  height?: number;
};

export type UploadScreenshotResult = {
  screenshotId: number;
  assetId: number | null;
  filePath: string;
};

export const uploadScreenshot = async (
  input: UploadScreenshotInput
): Promise<UploadScreenshotResult> => {
  const design = await db.query.evolutionDesigns.findFirst({
    where: eq(evolutionDesigns.id, input.designId),
  });

  if (!design) {
    throw new Error('Evolution design not found');
  }

  const request = await db.query.evolutionRequests.findFirst({
    where: eq(evolutionRequests.id, design.requestId),
  });

  if (!request || request.userId !== input.userId) {
    throw new Error('Evolution request not found');
  }

  const runId = await getOrCreateRunId(request.id, input.userId);
  const buffer = Buffer.from(input.dataBase64, 'base64');
  const hash = hashBuffer(buffer);
  const extension = extensionForContentType(input.contentType);
  const fileName = `${hash}${extension}`;
  const filePath = join(SCREENSHOT_DIR, fileName);

  await ensureScreenshotDir();

  try {
    await fs.access(filePath);
  } catch {
    await fs.writeFile(filePath, buffer);
  }

  let asset = await db.query.evolutionAssets.findFirst({
    where: and(
      eq(evolutionAssets.filePath, filePath),
      eq(evolutionAssets.userId, input.userId)
    ),
  });

  if (!asset) {
    const [inserted] = await db
      .insert(evolutionAssets)
      .values({
        runId,
        userId: input.userId,
        kind: input.kind ?? 'thumbnail',
        contentType: input.contentType,
        filePath,
        size: buffer.length,
      })
      .returning();
    asset = inserted;
  }

  const [screenshot] = await db
    .insert(evolutionScreenshots)
    .values({
      designId: input.designId,
      assetId: asset.id,
      kind: input.kind ?? 'thumbnail',
      width: input.width ?? null,
      height: input.height ?? null,
    })
    .returning();

  cacheService.setJson(
    `evolution-screenshot:${screenshot.id}`,
    {
      contentType: asset.contentType,
      dataBase64: input.dataBase64,
      width: screenshot.width ?? undefined,
      height: screenshot.height ?? undefined,
    },
    10 * 60 * 1000
  );

  return {
    screenshotId: screenshot.id,
    assetId: asset.id,
    filePath,
  };
};

export type ScreenshotPayload = {
  contentType: string;
  dataBase64: string;
  width?: number;
  height?: number;
};

export const getScreenshot = async (
  userId: string,
  screenshotId: number
): Promise<ScreenshotPayload | null> => {
  const cacheKey = `evolution-screenshot:${screenshotId}`;
  const cached = cacheService.getJson<ScreenshotPayload>(cacheKey);
  if (cached) return cached;

  const screenshot = await db.query.evolutionScreenshots.findFirst({
    where: eq(evolutionScreenshots.id, screenshotId),
  });

  if (!screenshot) return null;

  const design = await db.query.evolutionDesigns.findFirst({
    where: eq(evolutionDesigns.id, screenshot.designId),
  });

  if (!design) return null;

  const request = await db.query.evolutionRequests.findFirst({
    where: and(
      eq(evolutionRequests.id, design.requestId),
      eq(evolutionRequests.userId, userId)
    ),
  });

  if (!request || !screenshot.assetId) return null;

  const asset = await db.query.evolutionAssets.findFirst({
    where: and(
      eq(evolutionAssets.id, screenshot.assetId),
      eq(evolutionAssets.userId, userId)
    ),
  });

  if (!asset) return null;

  const file = await fs.readFile(asset.filePath);
  const payload: ScreenshotPayload = {
    contentType: asset.contentType,
    dataBase64: file.toString('base64'),
    width: screenshot.width ?? undefined,
    height: screenshot.height ?? undefined,
  };

  cacheService.setJson(cacheKey, payload, 10 * 60 * 1000);
  return payload;
};

export const getScreenshotUrl = (assetId: number) =>
  `/evolution-graph/assets/${assetId}`;
