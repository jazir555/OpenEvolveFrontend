import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
import { CredentialType } from '@bubblelab/shared-schemas';
import { ApifyBubble } from '../service-bubble/apify/apify.js';
// Unified Google Maps data types
const GoogleMapsReviewSchema = z.object({
    name: z.string().nullable().describe('Reviewer name'),
    rating: z.number().nullable().describe('Rating (1-5)'),
    text: z.string().nullable().describe('Review text'),
    publishedAtDate: z.string().nullable().describe('Publish date'),
    likesCount: z.number().nullable().describe('Number of likes'),
    responseFromOwnerText: z.string().nullable().describe('Owner response'),
});
const GoogleMapsPlaceSchema = z.object({
    title: z.string().nullable().describe('Place name'),
    placeId: z.string().nullable().describe('Place ID'),
    url: z.string().nullable().describe('Place URL'),
    address: z.string().nullable().describe('Full address'),
    category: z.string().nullable().describe('Primary category'),
    website: z.string().nullable().describe('Website URL'),
    phone: z.string().nullable().describe('Phone number'),
    rating: z.number().nullable().describe('Average rating'),
    reviewsCount: z.number().nullable().describe('Total reviews'),
    priceLevel: z.string().nullable().describe('Price level'),
    isAdvertisement: z.boolean().nullable().describe('Is sponsored'),
    location: z
        .object({
        lat: z.number().nullable(),
        lng: z.number().nullable(),
    })
        .nullable()
        .describe('Coordinates'),
    openingHours: z
        .array(z.object({
        day: z.string().nullable(),
        hours: z.string().nullable().or(z.array(z.string()).nullable()),
    }))
        .nullable()
        .describe('Opening hours'),
    reviews: z
        .array(GoogleMapsReviewSchema)
        .nullable()
        .describe('Recent reviews'),
    imageUrls: z.array(z.string()).nullable().describe('Image URLs'),
    additionalInfo: z
        .record(z.array(z.string()))
        .nullable()
        .describe('Additional attributes (accessibility, amenities, etc.)'),
});
const GoogleMapsToolParamsSchema = z.object({
    operation: z.enum(['search']).describe('Operation (only search supported)'),
    queries: z
        .array(z.string())
        .min(1)
        .describe('Search queries (e.g. "restaurants in NYC")'),
    location: z
        .string()
        .optional()
        .describe('Location to focus search (e.g. "New York, USA")'),
    limit: z
        .number()
        .min(1)
        .max(500)
        .default(20)
        .optional()
        .describe('Maximum number of places to scrape per query'),
    language: z.string().default('en').optional().describe('Result language'),
    credentials: z
        .record(z.nativeEnum(CredentialType), z.string())
        .optional()
        .describe('Required credentials (auto-injected)'),
});
const GoogleMapsToolResultSchema = z.object({
    operation: z.enum(['search']).describe('Operation performed'),
    places: z.array(GoogleMapsPlaceSchema).describe('Found places'),
    totalPlaces: z.number().describe('Total places found'),
    success: z.boolean().describe('Whether the operation was successful'),
    error: z.string().describe('Error message if operation failed'),
});
export class GoogleMapsTool extends ToolBubble {
    static bubbleName = 'google-maps-tool';
    static schema = GoogleMapsToolParamsSchema;
    static resultSchema = GoogleMapsToolResultSchema;
    static shortDescription = 'Scrape Google Maps business listings, reviews, and place data.';
    static longDescription = `
    Universal Google Maps scraping tool.
    
    Operations:
    - search: Find businesses and places by keyword and location
    
    Uses Apify's compass/crawler-google-places.
  `;
    static alias = 'maps';
    static type = 'tool';
    constructor(params = {
        operation: 'search',
        queries: ['restaurants in SF'],
        limit: 20,
    }, context) {
        super(params, context);
    }
    async performAction() {
        const credentials = this.params.credentials;
        if (!credentials || !credentials[CredentialType.APIFY_CRED]) {
            return this.createErrorResult('Google Maps scraping requires authentication. Please configure APIFY_CRED.');
        }
        try {
            const result = await this.runScraper();
            return {
                operation: 'search',
                places: result.places,
                totalPlaces: result.places.length,
                success: result.success,
                error: result.error,
            };
        }
        catch (error) {
            return this.createErrorResult(error instanceof Error ? error.message : 'Unknown error occurred');
        }
    }
    createErrorResult(errorMessage) {
        return {
            operation: 'search',
            places: [],
            totalPlaces: 0,
            success: false,
            error: errorMessage,
        };
    }
    async runScraper() {
        const { queries, location, limit, language } = this.params;
        const input = {
            searchStringsArray: queries,
            locationQuery: location,
            maxCrawledPlacesPerSearch: limit,
            language,
            // Default safer settings
            onlyDataFromSearchPage: false,
        };
        const scraper = new ApifyBubble({
            actorId: 'compass/crawler-google-places',
            input,
            waitForFinish: true,
            timeout: 240000, // 4 minutes, maps can be slow
            limit: limit,
            credentials: this.params.credentials,
        }, this.context, 'googleMapsScraper');
        const apifyResult = await scraper.action();
        if (!apifyResult.data.success) {
            return {
                places: [],
                success: false,
                error: apifyResult.data.error || 'Failed to scrape Google Maps',
            };
        }
        const items = apifyResult.data.items || [];
        const places = this.transformPlaces(items);
        return {
            places,
            success: true,
            error: '',
        };
    }
    transformPlaces(items) {
        return items.map((item) => ({
            title: item.title || null,
            placeId: item.placeId || null,
            url: item.url || null,
            address: item.address || null,
            category: item.categoryName || item.categories?.[0] || null,
            website: item.website || null,
            phone: item.phone || item.phoneUnformatted || null,
            rating: item.totalScore || null,
            reviewsCount: item.reviewsCount || null,
            priceLevel: item.price || null,
            isAdvertisement: item.isAdvertisement || null,
            location: item.location
                ? {
                    lat: item.location.lat || null,
                    lng: item.location.lng || null,
                }
                : null,
            openingHours: item.openingHours
                ? item.openingHours.map((h) => ({
                    day: h.day || null,
                    hours: h.hours || null, // Can be string or array per schema
                }))
                : null,
            reviews: null, // Reviews are not available in the current schema
            imageUrls: item.imageUrl ? [item.imageUrl] : null,
            additionalInfo: item.additionalInfo
                ? Object.fromEntries(Object.entries(item.additionalInfo).map(([key, value]) => [
                    key,
                    Array.isArray(value)
                        ? value
                            .flatMap((v) => typeof v === 'object' && v !== null
                            ? Object.keys(v).filter((k) => v[k] === true)
                            : [])
                            .filter((v) => v)
                        : [],
                ]))
                : null,
        }));
    }
}
//# sourceMappingURL=google-maps-tool.js.map