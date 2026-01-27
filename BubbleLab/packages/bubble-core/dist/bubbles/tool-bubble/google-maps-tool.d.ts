import { z } from 'zod';
import { ToolBubble } from '../../types/tool-bubble-class.js';
import type { BubbleContext } from '../../types/bubble.js';
import { CredentialType, type BubbleName } from '@bubblelab/shared-schemas';
declare const GoogleMapsToolParamsSchema: z.ZodObject<{
    operation: z.ZodEnum<["search"]>;
    queries: z.ZodArray<z.ZodString, "many">;
    location: z.ZodOptional<z.ZodString>;
    limit: z.ZodOptional<z.ZodDefault<z.ZodNumber>>;
    language: z.ZodOptional<z.ZodDefault<z.ZodString>>;
    credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
}, "strip", z.ZodTypeAny, {
    operation: "search";
    queries: string[];
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    limit?: number | undefined;
    location?: string | undefined;
    language?: string | undefined;
}, {
    operation: "search";
    queries: string[];
    credentials?: Partial<Record<CredentialType, string>> | undefined;
    limit?: number | undefined;
    location?: string | undefined;
    language?: string | undefined;
}>;
declare const GoogleMapsToolResultSchema: z.ZodObject<{
    operation: z.ZodEnum<["search"]>;
    places: z.ZodArray<z.ZodObject<{
        title: z.ZodNullable<z.ZodString>;
        placeId: z.ZodNullable<z.ZodString>;
        url: z.ZodNullable<z.ZodString>;
        address: z.ZodNullable<z.ZodString>;
        category: z.ZodNullable<z.ZodString>;
        website: z.ZodNullable<z.ZodString>;
        phone: z.ZodNullable<z.ZodString>;
        rating: z.ZodNullable<z.ZodNumber>;
        reviewsCount: z.ZodNullable<z.ZodNumber>;
        priceLevel: z.ZodNullable<z.ZodString>;
        isAdvertisement: z.ZodNullable<z.ZodBoolean>;
        location: z.ZodNullable<z.ZodObject<{
            lat: z.ZodNullable<z.ZodNumber>;
            lng: z.ZodNullable<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            lat: number | null;
            lng: number | null;
        }, {
            lat: number | null;
            lng: number | null;
        }>>;
        openingHours: z.ZodNullable<z.ZodArray<z.ZodObject<{
            day: z.ZodNullable<z.ZodString>;
            hours: z.ZodUnion<[z.ZodNullable<z.ZodString>, z.ZodNullable<z.ZodArray<z.ZodString, "many">>]>;
        }, "strip", z.ZodTypeAny, {
            day: string | null;
            hours: string | string[] | null;
        }, {
            day: string | null;
            hours: string | string[] | null;
        }>, "many">>;
        reviews: z.ZodNullable<z.ZodArray<z.ZodObject<{
            name: z.ZodNullable<z.ZodString>;
            rating: z.ZodNullable<z.ZodNumber>;
            text: z.ZodNullable<z.ZodString>;
            publishedAtDate: z.ZodNullable<z.ZodString>;
            likesCount: z.ZodNullable<z.ZodNumber>;
            responseFromOwnerText: z.ZodNullable<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            name: string | null;
            text: string | null;
            likesCount: number | null;
            rating: number | null;
            publishedAtDate: string | null;
            responseFromOwnerText: string | null;
        }, {
            name: string | null;
            text: string | null;
            likesCount: number | null;
            rating: number | null;
            publishedAtDate: string | null;
            responseFromOwnerText: string | null;
        }>, "many">>;
        imageUrls: z.ZodNullable<z.ZodArray<z.ZodString, "many">>;
        additionalInfo: z.ZodNullable<z.ZodRecord<z.ZodString, z.ZodArray<z.ZodString, "many">>>;
    }, "strip", z.ZodTypeAny, {
        title: string | null;
        url: string | null;
        phone: string | null;
        location: {
            lat: number | null;
            lng: number | null;
        } | null;
        rating: number | null;
        category: string | null;
        address: string | null;
        website: string | null;
        placeId: string | null;
        reviewsCount: number | null;
        openingHours: {
            day: string | null;
            hours: string | string[] | null;
        }[] | null;
        additionalInfo: Record<string, string[]> | null;
        isAdvertisement: boolean | null;
        priceLevel: string | null;
        reviews: {
            name: string | null;
            text: string | null;
            likesCount: number | null;
            rating: number | null;
            publishedAtDate: string | null;
            responseFromOwnerText: string | null;
        }[] | null;
        imageUrls: string[] | null;
    }, {
        title: string | null;
        url: string | null;
        phone: string | null;
        location: {
            lat: number | null;
            lng: number | null;
        } | null;
        rating: number | null;
        category: string | null;
        address: string | null;
        website: string | null;
        placeId: string | null;
        reviewsCount: number | null;
        openingHours: {
            day: string | null;
            hours: string | string[] | null;
        }[] | null;
        additionalInfo: Record<string, string[]> | null;
        isAdvertisement: boolean | null;
        priceLevel: string | null;
        reviews: {
            name: string | null;
            text: string | null;
            likesCount: number | null;
            rating: number | null;
            publishedAtDate: string | null;
            responseFromOwnerText: string | null;
        }[] | null;
        imageUrls: string[] | null;
    }>, "many">;
    totalPlaces: z.ZodNumber;
    success: z.ZodBoolean;
    error: z.ZodString;
}, "strip", z.ZodTypeAny, {
    error: string;
    success: boolean;
    operation: "search";
    places: {
        title: string | null;
        url: string | null;
        phone: string | null;
        location: {
            lat: number | null;
            lng: number | null;
        } | null;
        rating: number | null;
        category: string | null;
        address: string | null;
        website: string | null;
        placeId: string | null;
        reviewsCount: number | null;
        openingHours: {
            day: string | null;
            hours: string | string[] | null;
        }[] | null;
        additionalInfo: Record<string, string[]> | null;
        isAdvertisement: boolean | null;
        priceLevel: string | null;
        reviews: {
            name: string | null;
            text: string | null;
            likesCount: number | null;
            rating: number | null;
            publishedAtDate: string | null;
            responseFromOwnerText: string | null;
        }[] | null;
        imageUrls: string[] | null;
    }[];
    totalPlaces: number;
}, {
    error: string;
    success: boolean;
    operation: "search";
    places: {
        title: string | null;
        url: string | null;
        phone: string | null;
        location: {
            lat: number | null;
            lng: number | null;
        } | null;
        rating: number | null;
        category: string | null;
        address: string | null;
        website: string | null;
        placeId: string | null;
        reviewsCount: number | null;
        openingHours: {
            day: string | null;
            hours: string | string[] | null;
        }[] | null;
        additionalInfo: Record<string, string[]> | null;
        isAdvertisement: boolean | null;
        priceLevel: string | null;
        reviews: {
            name: string | null;
            text: string | null;
            likesCount: number | null;
            rating: number | null;
            publishedAtDate: string | null;
            responseFromOwnerText: string | null;
        }[] | null;
        imageUrls: string[] | null;
    }[];
    totalPlaces: number;
}>;
type GoogleMapsToolParams = z.output<typeof GoogleMapsToolParamsSchema>;
type GoogleMapsToolResult = z.output<typeof GoogleMapsToolResultSchema>;
type GoogleMapsToolParamsInput = z.input<typeof GoogleMapsToolParamsSchema>;
export declare class GoogleMapsTool extends ToolBubble<GoogleMapsToolParams, GoogleMapsToolResult> {
    static readonly bubbleName: BubbleName;
    static readonly schema: z.ZodObject<{
        operation: z.ZodEnum<["search"]>;
        queries: z.ZodArray<z.ZodString, "many">;
        location: z.ZodOptional<z.ZodString>;
        limit: z.ZodOptional<z.ZodDefault<z.ZodNumber>>;
        language: z.ZodOptional<z.ZodDefault<z.ZodString>>;
        credentials: z.ZodOptional<z.ZodRecord<z.ZodNativeEnum<typeof CredentialType>, z.ZodString>>;
    }, "strip", z.ZodTypeAny, {
        operation: "search";
        queries: string[];
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        limit?: number | undefined;
        location?: string | undefined;
        language?: string | undefined;
    }, {
        operation: "search";
        queries: string[];
        credentials?: Partial<Record<CredentialType, string>> | undefined;
        limit?: number | undefined;
        location?: string | undefined;
        language?: string | undefined;
    }>;
    static readonly resultSchema: z.ZodObject<{
        operation: z.ZodEnum<["search"]>;
        places: z.ZodArray<z.ZodObject<{
            title: z.ZodNullable<z.ZodString>;
            placeId: z.ZodNullable<z.ZodString>;
            url: z.ZodNullable<z.ZodString>;
            address: z.ZodNullable<z.ZodString>;
            category: z.ZodNullable<z.ZodString>;
            website: z.ZodNullable<z.ZodString>;
            phone: z.ZodNullable<z.ZodString>;
            rating: z.ZodNullable<z.ZodNumber>;
            reviewsCount: z.ZodNullable<z.ZodNumber>;
            priceLevel: z.ZodNullable<z.ZodString>;
            isAdvertisement: z.ZodNullable<z.ZodBoolean>;
            location: z.ZodNullable<z.ZodObject<{
                lat: z.ZodNullable<z.ZodNumber>;
                lng: z.ZodNullable<z.ZodNumber>;
            }, "strip", z.ZodTypeAny, {
                lat: number | null;
                lng: number | null;
            }, {
                lat: number | null;
                lng: number | null;
            }>>;
            openingHours: z.ZodNullable<z.ZodArray<z.ZodObject<{
                day: z.ZodNullable<z.ZodString>;
                hours: z.ZodUnion<[z.ZodNullable<z.ZodString>, z.ZodNullable<z.ZodArray<z.ZodString, "many">>]>;
            }, "strip", z.ZodTypeAny, {
                day: string | null;
                hours: string | string[] | null;
            }, {
                day: string | null;
                hours: string | string[] | null;
            }>, "many">>;
            reviews: z.ZodNullable<z.ZodArray<z.ZodObject<{
                name: z.ZodNullable<z.ZodString>;
                rating: z.ZodNullable<z.ZodNumber>;
                text: z.ZodNullable<z.ZodString>;
                publishedAtDate: z.ZodNullable<z.ZodString>;
                likesCount: z.ZodNullable<z.ZodNumber>;
                responseFromOwnerText: z.ZodNullable<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                name: string | null;
                text: string | null;
                likesCount: number | null;
                rating: number | null;
                publishedAtDate: string | null;
                responseFromOwnerText: string | null;
            }, {
                name: string | null;
                text: string | null;
                likesCount: number | null;
                rating: number | null;
                publishedAtDate: string | null;
                responseFromOwnerText: string | null;
            }>, "many">>;
            imageUrls: z.ZodNullable<z.ZodArray<z.ZodString, "many">>;
            additionalInfo: z.ZodNullable<z.ZodRecord<z.ZodString, z.ZodArray<z.ZodString, "many">>>;
        }, "strip", z.ZodTypeAny, {
            title: string | null;
            url: string | null;
            phone: string | null;
            location: {
                lat: number | null;
                lng: number | null;
            } | null;
            rating: number | null;
            category: string | null;
            address: string | null;
            website: string | null;
            placeId: string | null;
            reviewsCount: number | null;
            openingHours: {
                day: string | null;
                hours: string | string[] | null;
            }[] | null;
            additionalInfo: Record<string, string[]> | null;
            isAdvertisement: boolean | null;
            priceLevel: string | null;
            reviews: {
                name: string | null;
                text: string | null;
                likesCount: number | null;
                rating: number | null;
                publishedAtDate: string | null;
                responseFromOwnerText: string | null;
            }[] | null;
            imageUrls: string[] | null;
        }, {
            title: string | null;
            url: string | null;
            phone: string | null;
            location: {
                lat: number | null;
                lng: number | null;
            } | null;
            rating: number | null;
            category: string | null;
            address: string | null;
            website: string | null;
            placeId: string | null;
            reviewsCount: number | null;
            openingHours: {
                day: string | null;
                hours: string | string[] | null;
            }[] | null;
            additionalInfo: Record<string, string[]> | null;
            isAdvertisement: boolean | null;
            priceLevel: string | null;
            reviews: {
                name: string | null;
                text: string | null;
                likesCount: number | null;
                rating: number | null;
                publishedAtDate: string | null;
                responseFromOwnerText: string | null;
            }[] | null;
            imageUrls: string[] | null;
        }>, "many">;
        totalPlaces: z.ZodNumber;
        success: z.ZodBoolean;
        error: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        error: string;
        success: boolean;
        operation: "search";
        places: {
            title: string | null;
            url: string | null;
            phone: string | null;
            location: {
                lat: number | null;
                lng: number | null;
            } | null;
            rating: number | null;
            category: string | null;
            address: string | null;
            website: string | null;
            placeId: string | null;
            reviewsCount: number | null;
            openingHours: {
                day: string | null;
                hours: string | string[] | null;
            }[] | null;
            additionalInfo: Record<string, string[]> | null;
            isAdvertisement: boolean | null;
            priceLevel: string | null;
            reviews: {
                name: string | null;
                text: string | null;
                likesCount: number | null;
                rating: number | null;
                publishedAtDate: string | null;
                responseFromOwnerText: string | null;
            }[] | null;
            imageUrls: string[] | null;
        }[];
        totalPlaces: number;
    }, {
        error: string;
        success: boolean;
        operation: "search";
        places: {
            title: string | null;
            url: string | null;
            phone: string | null;
            location: {
                lat: number | null;
                lng: number | null;
            } | null;
            rating: number | null;
            category: string | null;
            address: string | null;
            website: string | null;
            placeId: string | null;
            reviewsCount: number | null;
            openingHours: {
                day: string | null;
                hours: string | string[] | null;
            }[] | null;
            additionalInfo: Record<string, string[]> | null;
            isAdvertisement: boolean | null;
            priceLevel: string | null;
            reviews: {
                name: string | null;
                text: string | null;
                likesCount: number | null;
                rating: number | null;
                publishedAtDate: string | null;
                responseFromOwnerText: string | null;
            }[] | null;
            imageUrls: string[] | null;
        }[];
        totalPlaces: number;
    }>;
    static readonly shortDescription = "Scrape Google Maps business listings, reviews, and place data.";
    static readonly longDescription = "\n    Universal Google Maps scraping tool.\n    \n    Operations:\n    - search: Find businesses and places by keyword and location\n    \n    Uses Apify's compass/crawler-google-places.\n  ";
    static readonly alias = "maps";
    static readonly type = "tool";
    constructor(params?: GoogleMapsToolParamsInput, context?: BubbleContext);
    performAction(): Promise<GoogleMapsToolResult>;
    private createErrorResult;
    private runScraper;
    private transformPlaces;
}
export {};
//# sourceMappingURL=google-maps-tool.d.ts.map