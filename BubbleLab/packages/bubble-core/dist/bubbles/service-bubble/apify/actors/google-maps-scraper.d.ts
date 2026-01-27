import { z } from 'zod';
export declare const GoogleMapsScraperInputSchema: z.ZodObject<{
    searchStringsArray: z.ZodArray<z.ZodString, "many">;
    locationQuery: z.ZodOptional<z.ZodString>;
    maxCrawledPlacesPerSearch: z.ZodOptional<z.ZodDefault<z.ZodNumber>>;
    language: z.ZodOptional<z.ZodDefault<z.ZodString>>;
    onlyDataFromSearchPage: z.ZodOptional<z.ZodDefault<z.ZodBoolean>>;
}, "strip", z.ZodTypeAny, {
    searchStringsArray: string[];
    language?: string | undefined;
    locationQuery?: string | undefined;
    maxCrawledPlacesPerSearch?: number | undefined;
    onlyDataFromSearchPage?: boolean | undefined;
}, {
    searchStringsArray: string[];
    language?: string | undefined;
    locationQuery?: string | undefined;
    maxCrawledPlacesPerSearch?: number | undefined;
    onlyDataFromSearchPage?: boolean | undefined;
}>;
export declare const GoogleMapsPlaceSchema: z.ZodObject<{
    title: z.ZodOptional<z.ZodString>;
    description: z.ZodOptional<z.ZodString>;
    price: z.ZodOptional<z.ZodNullable<z.ZodString>>;
    categoryName: z.ZodOptional<z.ZodString>;
    address: z.ZodOptional<z.ZodString>;
    neighborhood: z.ZodOptional<z.ZodNullable<z.ZodString>>;
    street: z.ZodOptional<z.ZodNullable<z.ZodString>>;
    city: z.ZodOptional<z.ZodString>;
    postalCode: z.ZodOptional<z.ZodNullable<z.ZodString>>;
    state: z.ZodOptional<z.ZodString>;
    countryCode: z.ZodOptional<z.ZodString>;
    website: z.ZodOptional<z.ZodString>;
    phone: z.ZodOptional<z.ZodString>;
    phoneUnformatted: z.ZodOptional<z.ZodString>;
    claimThisBusiness: z.ZodOptional<z.ZodBoolean>;
    location: z.ZodOptional<z.ZodObject<{
        lat: z.ZodNumber;
        lng: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        lat: number;
        lng: number;
    }, {
        lat: number;
        lng: number;
    }>>;
    locatedIn: z.ZodOptional<z.ZodString>;
    totalScore: z.ZodOptional<z.ZodNumber>;
    permanentlyClosed: z.ZodOptional<z.ZodBoolean>;
    temporarilyClosed: z.ZodOptional<z.ZodBoolean>;
    placeId: z.ZodOptional<z.ZodString>;
    categories: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    fid: z.ZodOptional<z.ZodString>;
    cid: z.ZodOptional<z.ZodString>;
    reviewsCount: z.ZodOptional<z.ZodNullable<z.ZodNumber>>;
    reviewsDistribution: z.ZodOptional<z.ZodObject<{
        oneStar: z.ZodOptional<z.ZodNumber>;
        twoStar: z.ZodOptional<z.ZodNumber>;
        threeStar: z.ZodOptional<z.ZodNumber>;
        fourStar: z.ZodOptional<z.ZodNumber>;
        fiveStar: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        oneStar?: number | undefined;
        twoStar?: number | undefined;
        threeStar?: number | undefined;
        fourStar?: number | undefined;
        fiveStar?: number | undefined;
    }, {
        oneStar?: number | undefined;
        twoStar?: number | undefined;
        threeStar?: number | undefined;
        fourStar?: number | undefined;
        fiveStar?: number | undefined;
    }>>;
    imagesCount: z.ZodOptional<z.ZodNumber>;
    imageCategories: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    scrapedAt: z.ZodOptional<z.ZodString>;
    googleFoodUrl: z.ZodOptional<z.ZodNullable<z.ZodString>>;
    hotelAds: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
    openingHours: z.ZodOptional<z.ZodArray<z.ZodObject<{
        day: z.ZodString;
        hours: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        day: string;
        hours: string;
    }, {
        day: string;
        hours: string;
    }>, "many">>;
    additionalOpeningHours: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodArray<z.ZodObject<{
        day: z.ZodString;
        hours: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        day: string;
        hours: string;
    }, {
        day: string;
        hours: string;
    }>, "many">>>;
    peopleAlsoSearch: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    placesTags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    reviewsTags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    additionalInfo: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodArray<z.ZodRecord<z.ZodString, z.ZodBoolean>, "many">>>;
    gasPrices: z.ZodOptional<z.ZodArray<z.ZodUnknown, "many">>;
    url: z.ZodOptional<z.ZodString>;
    searchPageUrl: z.ZodOptional<z.ZodString>;
    searchString: z.ZodOptional<z.ZodString>;
    language: z.ZodOptional<z.ZodString>;
    rank: z.ZodOptional<z.ZodNumber>;
    isAdvertisement: z.ZodOptional<z.ZodBoolean>;
    imageUrl: z.ZodOptional<z.ZodString>;
    kgmid: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    description?: string | undefined;
    title?: string | undefined;
    url?: string | undefined;
    phone?: string | undefined;
    location?: {
        lat: number;
        lng: number;
    } | undefined;
    language?: string | undefined;
    postalCode?: string | null | undefined;
    price?: string | null | undefined;
    categoryName?: string | undefined;
    address?: string | undefined;
    neighborhood?: string | null | undefined;
    street?: string | null | undefined;
    city?: string | undefined;
    state?: string | undefined;
    countryCode?: string | undefined;
    website?: string | undefined;
    phoneUnformatted?: string | undefined;
    claimThisBusiness?: boolean | undefined;
    locatedIn?: string | undefined;
    totalScore?: number | undefined;
    permanentlyClosed?: boolean | undefined;
    temporarilyClosed?: boolean | undefined;
    placeId?: string | undefined;
    categories?: string[] | undefined;
    fid?: string | undefined;
    cid?: string | undefined;
    reviewsCount?: number | null | undefined;
    reviewsDistribution?: {
        oneStar?: number | undefined;
        twoStar?: number | undefined;
        threeStar?: number | undefined;
        fourStar?: number | undefined;
        fiveStar?: number | undefined;
    } | undefined;
    imagesCount?: number | undefined;
    imageCategories?: string[] | undefined;
    scrapedAt?: string | undefined;
    googleFoodUrl?: string | null | undefined;
    hotelAds?: unknown[] | undefined;
    openingHours?: {
        day: string;
        hours: string;
    }[] | undefined;
    additionalOpeningHours?: Record<string, {
        day: string;
        hours: string;
    }[]> | undefined;
    peopleAlsoSearch?: string[] | undefined;
    placesTags?: string[] | undefined;
    reviewsTags?: string[] | undefined;
    additionalInfo?: Record<string, Record<string, boolean>[]> | undefined;
    gasPrices?: unknown[] | undefined;
    searchPageUrl?: string | undefined;
    searchString?: string | undefined;
    rank?: number | undefined;
    isAdvertisement?: boolean | undefined;
    imageUrl?: string | undefined;
    kgmid?: string | undefined;
}, {
    description?: string | undefined;
    title?: string | undefined;
    url?: string | undefined;
    phone?: string | undefined;
    location?: {
        lat: number;
        lng: number;
    } | undefined;
    language?: string | undefined;
    postalCode?: string | null | undefined;
    price?: string | null | undefined;
    categoryName?: string | undefined;
    address?: string | undefined;
    neighborhood?: string | null | undefined;
    street?: string | null | undefined;
    city?: string | undefined;
    state?: string | undefined;
    countryCode?: string | undefined;
    website?: string | undefined;
    phoneUnformatted?: string | undefined;
    claimThisBusiness?: boolean | undefined;
    locatedIn?: string | undefined;
    totalScore?: number | undefined;
    permanentlyClosed?: boolean | undefined;
    temporarilyClosed?: boolean | undefined;
    placeId?: string | undefined;
    categories?: string[] | undefined;
    fid?: string | undefined;
    cid?: string | undefined;
    reviewsCount?: number | null | undefined;
    reviewsDistribution?: {
        oneStar?: number | undefined;
        twoStar?: number | undefined;
        threeStar?: number | undefined;
        fourStar?: number | undefined;
        fiveStar?: number | undefined;
    } | undefined;
    imagesCount?: number | undefined;
    imageCategories?: string[] | undefined;
    scrapedAt?: string | undefined;
    googleFoodUrl?: string | null | undefined;
    hotelAds?: unknown[] | undefined;
    openingHours?: {
        day: string;
        hours: string;
    }[] | undefined;
    additionalOpeningHours?: Record<string, {
        day: string;
        hours: string;
    }[]> | undefined;
    peopleAlsoSearch?: string[] | undefined;
    placesTags?: string[] | undefined;
    reviewsTags?: string[] | undefined;
    additionalInfo?: Record<string, Record<string, boolean>[]> | undefined;
    gasPrices?: unknown[] | undefined;
    searchPageUrl?: string | undefined;
    searchString?: string | undefined;
    rank?: number | undefined;
    isAdvertisement?: boolean | undefined;
    imageUrl?: string | undefined;
    kgmid?: string | undefined;
}>;
export type GoogleMapsScraperInput = z.output<typeof GoogleMapsScraperInputSchema>;
export type GoogleMapsPlace = z.output<typeof GoogleMapsPlaceSchema>;
//# sourceMappingURL=google-maps-scraper.d.ts.map