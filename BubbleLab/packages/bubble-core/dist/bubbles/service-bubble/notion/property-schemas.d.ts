import { z } from 'zod';
declare const ColorEnum: z.ZodEnum<["default", "gray", "brown", "orange", "yellow", "green", "blue", "purple", "pink", "red"]>;
declare const CheckboxPropertySchema: z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    type: z.ZodLiteral<"checkbox">;
    checkbox: z.ZodObject<{}, "strip", z.ZodTypeAny, {}, {}>;
}, "strip", z.ZodTypeAny, {
    type: "checkbox";
    name: string;
    id: string;
    checkbox: {};
    description?: string | undefined;
}, {
    type: "checkbox";
    name: string;
    id: string;
    checkbox: {};
    description?: string | undefined;
}>;
declare const CreatedByPropertySchema: z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    type: z.ZodLiteral<"created_by">;
    created_by: z.ZodObject<{}, "strip", z.ZodTypeAny, {}, {}>;
}, "strip", z.ZodTypeAny, {
    type: "created_by";
    name: string;
    id: string;
    created_by: {};
    description?: string | undefined;
}, {
    type: "created_by";
    name: string;
    id: string;
    created_by: {};
    description?: string | undefined;
}>;
declare const CreatedTimePropertySchema: z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    type: z.ZodLiteral<"created_time">;
    created_time: z.ZodObject<{}, "strip", z.ZodTypeAny, {}, {}>;
}, "strip", z.ZodTypeAny, {
    type: "created_time";
    name: string;
    id: string;
    created_time: {};
    description?: string | undefined;
}, {
    type: "created_time";
    name: string;
    id: string;
    created_time: {};
    description?: string | undefined;
}>;
declare const DatePropertySchema: z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    type: z.ZodLiteral<"date">;
    date: z.ZodObject<{}, "strip", z.ZodTypeAny, {}, {}>;
}, "strip", z.ZodTypeAny, {
    type: "date";
    date: {};
    name: string;
    id: string;
    description?: string | undefined;
}, {
    type: "date";
    date: {};
    name: string;
    id: string;
    description?: string | undefined;
}>;
declare const EmailPropertySchema: z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    type: z.ZodLiteral<"email">;
    email: z.ZodObject<{}, "strip", z.ZodTypeAny, {}, {}>;
}, "strip", z.ZodTypeAny, {
    type: "email";
    name: string;
    email: {};
    id: string;
    description?: string | undefined;
}, {
    type: "email";
    name: string;
    email: {};
    id: string;
    description?: string | undefined;
}>;
declare const FilesPropertySchema: z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    type: z.ZodLiteral<"files">;
    files: z.ZodObject<{}, "strip", z.ZodTypeAny, {}, {}>;
}, "strip", z.ZodTypeAny, {
    type: "files";
    name: string;
    id: string;
    files: {};
    description?: string | undefined;
}, {
    type: "files";
    name: string;
    id: string;
    files: {};
    description?: string | undefined;
}>;
declare const FormulaPropertySchema: z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    type: z.ZodLiteral<"formula">;
    formula: z.ZodObject<{
        expression: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        expression: string;
    }, {
        expression: string;
    }>;
}, "strip", z.ZodTypeAny, {
    type: "formula";
    name: string;
    id: string;
    formula: {
        expression: string;
    };
    description?: string | undefined;
}, {
    type: "formula";
    name: string;
    id: string;
    formula: {
        expression: string;
    };
    description?: string | undefined;
}>;
declare const LastEditedByPropertySchema: z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    type: z.ZodLiteral<"last_edited_by">;
    last_edited_by: z.ZodObject<{}, "strip", z.ZodTypeAny, {}, {}>;
}, "strip", z.ZodTypeAny, {
    type: "last_edited_by";
    name: string;
    id: string;
    last_edited_by: {};
    description?: string | undefined;
}, {
    type: "last_edited_by";
    name: string;
    id: string;
    last_edited_by: {};
    description?: string | undefined;
}>;
declare const LastEditedTimePropertySchema: z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    type: z.ZodLiteral<"last_edited_time">;
    last_edited_time: z.ZodObject<{}, "strip", z.ZodTypeAny, {}, {}>;
}, "strip", z.ZodTypeAny, {
    type: "last_edited_time";
    name: string;
    id: string;
    last_edited_time: {};
    description?: string | undefined;
}, {
    type: "last_edited_time";
    name: string;
    id: string;
    last_edited_time: {};
    description?: string | undefined;
}>;
declare const MultiSelectOptionSchema: z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    color: z.ZodEnum<["default", "gray", "brown", "orange", "yellow", "green", "blue", "purple", "pink", "red"]>;
}, "strip", z.ZodTypeAny, {
    name: string;
    color: "default" | "gray" | "brown" | "orange" | "yellow" | "green" | "blue" | "purple" | "pink" | "red";
    id: string;
}, {
    name: string;
    color: "default" | "gray" | "brown" | "orange" | "yellow" | "green" | "blue" | "purple" | "pink" | "red";
    id: string;
}>;
declare const MultiSelectPropertySchema: z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    type: z.ZodLiteral<"multi_select">;
    multi_select: z.ZodObject<{
        options: z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            name: z.ZodString;
            color: z.ZodEnum<["default", "gray", "brown", "orange", "yellow", "green", "blue", "purple", "pink", "red"]>;
        }, "strip", z.ZodTypeAny, {
            name: string;
            color: "default" | "gray" | "brown" | "orange" | "yellow" | "green" | "blue" | "purple" | "pink" | "red";
            id: string;
        }, {
            name: string;
            color: "default" | "gray" | "brown" | "orange" | "yellow" | "green" | "blue" | "purple" | "pink" | "red";
            id: string;
        }>, "many">;
    }, "strip", z.ZodTypeAny, {
        options: {
            name: string;
            color: "default" | "gray" | "brown" | "orange" | "yellow" | "green" | "blue" | "purple" | "pink" | "red";
            id: string;
        }[];
    }, {
        options: {
            name: string;
            color: "default" | "gray" | "brown" | "orange" | "yellow" | "green" | "blue" | "purple" | "pink" | "red";
            id: string;
        }[];
    }>;
}, "strip", z.ZodTypeAny, {
    type: "multi_select";
    name: string;
    id: string;
    multi_select: {
        options: {
            name: string;
            color: "default" | "gray" | "brown" | "orange" | "yellow" | "green" | "blue" | "purple" | "pink" | "red";
            id: string;
        }[];
    };
    description?: string | undefined;
}, {
    type: "multi_select";
    name: string;
    id: string;
    multi_select: {
        options: {
            name: string;
            color: "default" | "gray" | "brown" | "orange" | "yellow" | "green" | "blue" | "purple" | "pink" | "red";
            id: string;
        }[];
    };
    description?: string | undefined;
}>;
declare const NumberFormatEnum: z.ZodEnum<["number", "number_with_commas", "percent", "dollar", "canadian_dollar", "singapore_dollar", "euro", "pound", "yen", "ruble", "rupee", "won", "yuan", "real", "lira", "rupiah", "franc", "hong_kong_dollar", "new_zealand_dollar", "krona", "norwegian_krone", "mexican_peso", "rand", "new_taiwan_dollar", "danish_krone", "zloty", "baht", "forint", "koruna", "shekel", "chilean_peso", "philippine_peso", "dirham", "colombian_peso", "riyal", "ringgit", "leu", "argentine_peso", "uruguayan_peso", "peruvian_sol"]>;
declare const NumberPropertySchema: z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    type: z.ZodLiteral<"number">;
    number: z.ZodObject<{
        format: z.ZodEnum<["number", "number_with_commas", "percent", "dollar", "canadian_dollar", "singapore_dollar", "euro", "pound", "yen", "ruble", "rupee", "won", "yuan", "real", "lira", "rupiah", "franc", "hong_kong_dollar", "new_zealand_dollar", "krona", "norwegian_krone", "mexican_peso", "rand", "new_taiwan_dollar", "danish_krone", "zloty", "baht", "forint", "koruna", "shekel", "chilean_peso", "philippine_peso", "dirham", "colombian_peso", "riyal", "ringgit", "leu", "argentine_peso", "uruguayan_peso", "peruvian_sol"]>;
    }, "strip", z.ZodTypeAny, {
        format: "number" | "percent" | "number_with_commas" | "dollar" | "canadian_dollar" | "singapore_dollar" | "euro" | "pound" | "yen" | "ruble" | "rupee" | "won" | "yuan" | "real" | "lira" | "rupiah" | "franc" | "hong_kong_dollar" | "new_zealand_dollar" | "krona" | "norwegian_krone" | "mexican_peso" | "rand" | "new_taiwan_dollar" | "danish_krone" | "zloty" | "baht" | "forint" | "koruna" | "shekel" | "chilean_peso" | "philippine_peso" | "dirham" | "colombian_peso" | "riyal" | "ringgit" | "leu" | "argentine_peso" | "uruguayan_peso" | "peruvian_sol";
    }, {
        format: "number" | "percent" | "number_with_commas" | "dollar" | "canadian_dollar" | "singapore_dollar" | "euro" | "pound" | "yen" | "ruble" | "rupee" | "won" | "yuan" | "real" | "lira" | "rupiah" | "franc" | "hong_kong_dollar" | "new_zealand_dollar" | "krona" | "norwegian_krone" | "mexican_peso" | "rand" | "new_taiwan_dollar" | "danish_krone" | "zloty" | "baht" | "forint" | "koruna" | "shekel" | "chilean_peso" | "philippine_peso" | "dirham" | "colombian_peso" | "riyal" | "ringgit" | "leu" | "argentine_peso" | "uruguayan_peso" | "peruvian_sol";
    }>;
}, "strip", z.ZodTypeAny, {
    number: {
        format: "number" | "percent" | "number_with_commas" | "dollar" | "canadian_dollar" | "singapore_dollar" | "euro" | "pound" | "yen" | "ruble" | "rupee" | "won" | "yuan" | "real" | "lira" | "rupiah" | "franc" | "hong_kong_dollar" | "new_zealand_dollar" | "krona" | "norwegian_krone" | "mexican_peso" | "rand" | "new_taiwan_dollar" | "danish_krone" | "zloty" | "baht" | "forint" | "koruna" | "shekel" | "chilean_peso" | "philippine_peso" | "dirham" | "colombian_peso" | "riyal" | "ringgit" | "leu" | "argentine_peso" | "uruguayan_peso" | "peruvian_sol";
    };
    type: "number";
    name: string;
    id: string;
    description?: string | undefined;
}, {
    number: {
        format: "number" | "percent" | "number_with_commas" | "dollar" | "canadian_dollar" | "singapore_dollar" | "euro" | "pound" | "yen" | "ruble" | "rupee" | "won" | "yuan" | "real" | "lira" | "rupiah" | "franc" | "hong_kong_dollar" | "new_zealand_dollar" | "krona" | "norwegian_krone" | "mexican_peso" | "rand" | "new_taiwan_dollar" | "danish_krone" | "zloty" | "baht" | "forint" | "koruna" | "shekel" | "chilean_peso" | "philippine_peso" | "dirham" | "colombian_peso" | "riyal" | "ringgit" | "leu" | "argentine_peso" | "uruguayan_peso" | "peruvian_sol";
    };
    type: "number";
    name: string;
    id: string;
    description?: string | undefined;
}>;
declare const PeoplePropertySchema: z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    type: z.ZodLiteral<"people">;
    people: z.ZodObject<{}, "strip", z.ZodTypeAny, {}, {}>;
}, "strip", z.ZodTypeAny, {
    type: "people";
    name: string;
    id: string;
    people: {};
    description?: string | undefined;
}, {
    type: "people";
    name: string;
    id: string;
    people: {};
    description?: string | undefined;
}>;
declare const PhoneNumberPropertySchema: z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    type: z.ZodLiteral<"phone_number">;
    phone_number: z.ZodObject<{}, "strip", z.ZodTypeAny, {}, {}>;
}, "strip", z.ZodTypeAny, {
    type: "phone_number";
    name: string;
    id: string;
    phone_number: {};
    description?: string | undefined;
}, {
    type: "phone_number";
    name: string;
    id: string;
    phone_number: {};
    description?: string | undefined;
}>;
declare const PlacePropertySchema: z.ZodObject<{
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    type: z.ZodLiteral<"place">;
    place: z.ZodObject<{}, "strip", z.ZodTypeAny, {}, {}>;
}, "strip", z.ZodTypeAny, {
    type: "place";
    name: string;
    place: {};
    description?: string | undefined;
}, {
    type: "place";
    name: string;
    place: {};
    description?: string | undefined;
}>;
declare const RelationPropertySchema: z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    type: z.ZodLiteral<"relation">;
    relation: z.ZodObject<{
        data_source_id: z.ZodString;
        dual_property: z.ZodOptional<z.ZodObject<{
            synced_property_id: z.ZodString;
            synced_property_name: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            synced_property_id: string;
            synced_property_name: string;
        }, {
            synced_property_id: string;
            synced_property_name: string;
        }>>;
    }, "strip", z.ZodTypeAny, {
        data_source_id: string;
        dual_property?: {
            synced_property_id: string;
            synced_property_name: string;
        } | undefined;
    }, {
        data_source_id: string;
        dual_property?: {
            synced_property_id: string;
            synced_property_name: string;
        } | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    type: "relation";
    name: string;
    id: string;
    relation: {
        data_source_id: string;
        dual_property?: {
            synced_property_id: string;
            synced_property_name: string;
        } | undefined;
    };
    description?: string | undefined;
}, {
    type: "relation";
    name: string;
    id: string;
    relation: {
        data_source_id: string;
        dual_property?: {
            synced_property_id: string;
            synced_property_name: string;
        } | undefined;
    };
    description?: string | undefined;
}>;
declare const RichTextPropertySchema: z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    type: z.ZodLiteral<"rich_text">;
    rich_text: z.ZodObject<{}, "strip", z.ZodTypeAny, {}, {}>;
}, "strip", z.ZodTypeAny, {
    type: "rich_text";
    name: string;
    id: string;
    rich_text: {};
    description?: string | undefined;
}, {
    type: "rich_text";
    name: string;
    id: string;
    rich_text: {};
    description?: string | undefined;
}>;
declare const RollupFunctionEnum: z.ZodEnum<["average", "checked", "count_per_group", "count", "count_values", "date_range", "earliest_date", "empty", "latest_date", "max", "median", "min", "not_empty", "percent_checked", "percent_empty", "percent_not_empty", "percent_per_group", "percent_unchecked", "range", "unchecked", "unique", "show_original", "show_unique", "sum"]>;
declare const RollupPropertySchema: z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    type: z.ZodLiteral<"rollup">;
    rollup: z.ZodObject<{
        function: z.ZodEnum<["average", "checked", "count_per_group", "count", "count_values", "date_range", "earliest_date", "empty", "latest_date", "max", "median", "min", "not_empty", "percent_checked", "percent_empty", "percent_not_empty", "percent_per_group", "percent_unchecked", "range", "unchecked", "unique", "show_original", "show_unique", "sum"]>;
        relation_property_id: z.ZodString;
        relation_property_name: z.ZodString;
        rollup_property_id: z.ZodString;
        rollup_property_name: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        function: "min" | "max" | "count" | "range" | "average" | "checked" | "count_per_group" | "count_values" | "date_range" | "earliest_date" | "empty" | "latest_date" | "median" | "not_empty" | "percent_checked" | "percent_empty" | "percent_not_empty" | "percent_per_group" | "percent_unchecked" | "unchecked" | "unique" | "show_original" | "show_unique" | "sum";
        relation_property_id: string;
        relation_property_name: string;
        rollup_property_id: string;
        rollup_property_name: string;
    }, {
        function: "min" | "max" | "count" | "range" | "average" | "checked" | "count_per_group" | "count_values" | "date_range" | "earliest_date" | "empty" | "latest_date" | "median" | "not_empty" | "percent_checked" | "percent_empty" | "percent_not_empty" | "percent_per_group" | "percent_unchecked" | "unchecked" | "unique" | "show_original" | "show_unique" | "sum";
        relation_property_id: string;
        relation_property_name: string;
        rollup_property_id: string;
        rollup_property_name: string;
    }>;
}, "strip", z.ZodTypeAny, {
    type: "rollup";
    name: string;
    id: string;
    rollup: {
        function: "min" | "max" | "count" | "range" | "average" | "checked" | "count_per_group" | "count_values" | "date_range" | "earliest_date" | "empty" | "latest_date" | "median" | "not_empty" | "percent_checked" | "percent_empty" | "percent_not_empty" | "percent_per_group" | "percent_unchecked" | "unchecked" | "unique" | "show_original" | "show_unique" | "sum";
        relation_property_id: string;
        relation_property_name: string;
        rollup_property_id: string;
        rollup_property_name: string;
    };
    description?: string | undefined;
}, {
    type: "rollup";
    name: string;
    id: string;
    rollup: {
        function: "min" | "max" | "count" | "range" | "average" | "checked" | "count_per_group" | "count_values" | "date_range" | "earliest_date" | "empty" | "latest_date" | "median" | "not_empty" | "percent_checked" | "percent_empty" | "percent_not_empty" | "percent_per_group" | "percent_unchecked" | "unchecked" | "unique" | "show_original" | "show_unique" | "sum";
        relation_property_id: string;
        relation_property_name: string;
        rollup_property_id: string;
        rollup_property_name: string;
    };
    description?: string | undefined;
}>;
declare const SelectOptionSchema: z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    color: z.ZodEnum<["default", "gray", "brown", "orange", "yellow", "green", "blue", "purple", "pink", "red"]>;
}, "strip", z.ZodTypeAny, {
    name: string;
    color: "default" | "gray" | "brown" | "orange" | "yellow" | "green" | "blue" | "purple" | "pink" | "red";
    id: string;
}, {
    name: string;
    color: "default" | "gray" | "brown" | "orange" | "yellow" | "green" | "blue" | "purple" | "pink" | "red";
    id: string;
}>;
declare const SelectPropertySchema: z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    type: z.ZodLiteral<"select">;
    select: z.ZodObject<{
        options: z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            name: z.ZodString;
            color: z.ZodEnum<["default", "gray", "brown", "orange", "yellow", "green", "blue", "purple", "pink", "red"]>;
        }, "strip", z.ZodTypeAny, {
            name: string;
            color: "default" | "gray" | "brown" | "orange" | "yellow" | "green" | "blue" | "purple" | "pink" | "red";
            id: string;
        }, {
            name: string;
            color: "default" | "gray" | "brown" | "orange" | "yellow" | "green" | "blue" | "purple" | "pink" | "red";
            id: string;
        }>, "many">;
    }, "strip", z.ZodTypeAny, {
        options: {
            name: string;
            color: "default" | "gray" | "brown" | "orange" | "yellow" | "green" | "blue" | "purple" | "pink" | "red";
            id: string;
        }[];
    }, {
        options: {
            name: string;
            color: "default" | "gray" | "brown" | "orange" | "yellow" | "green" | "blue" | "purple" | "pink" | "red";
            id: string;
        }[];
    }>;
}, "strip", z.ZodTypeAny, {
    type: "select";
    name: string;
    id: string;
    select: {
        options: {
            name: string;
            color: "default" | "gray" | "brown" | "orange" | "yellow" | "green" | "blue" | "purple" | "pink" | "red";
            id: string;
        }[];
    };
    description?: string | undefined;
}, {
    type: "select";
    name: string;
    id: string;
    select: {
        options: {
            name: string;
            color: "default" | "gray" | "brown" | "orange" | "yellow" | "green" | "blue" | "purple" | "pink" | "red";
            id: string;
        }[];
    };
    description?: string | undefined;
}>;
declare const TitlePropertySchema: z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    type: z.ZodLiteral<"title">;
    title: z.ZodObject<{}, "strip", z.ZodTypeAny, {}, {}>;
}, "strip", z.ZodTypeAny, {
    type: "title";
    title: {};
    name: string;
    id: string;
    description?: string | undefined;
}, {
    type: "title";
    title: {};
    name: string;
    id: string;
    description?: string | undefined;
}>;
declare const URLPropertySchema: z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    type: z.ZodLiteral<"url">;
    url: z.ZodObject<{}, "strip", z.ZodTypeAny, {}, {}>;
}, "strip", z.ZodTypeAny, {
    type: "url";
    name: string;
    url: {};
    id: string;
    description?: string | undefined;
}, {
    type: "url";
    name: string;
    url: {};
    id: string;
    description?: string | undefined;
}>;
declare const UniqueIDPropertySchema: z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    type: z.ZodLiteral<"unique_id">;
    unique_id: z.ZodObject<{
        prefix: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        prefix?: string | undefined;
    }, {
        prefix?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    type: "unique_id";
    name: string;
    id: string;
    unique_id: {
        prefix?: string | undefined;
    };
    description?: string | undefined;
}, {
    type: "unique_id";
    name: string;
    id: string;
    unique_id: {
        prefix?: string | undefined;
    };
    description?: string | undefined;
}>;
export declare const DataSourcePropertySchema: z.ZodDiscriminatedUnion<"type", [z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    type: z.ZodLiteral<"checkbox">;
    checkbox: z.ZodObject<{}, "strip", z.ZodTypeAny, {}, {}>;
}, "strip", z.ZodTypeAny, {
    type: "checkbox";
    name: string;
    id: string;
    checkbox: {};
    description?: string | undefined;
}, {
    type: "checkbox";
    name: string;
    id: string;
    checkbox: {};
    description?: string | undefined;
}>, z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    type: z.ZodLiteral<"created_by">;
    created_by: z.ZodObject<{}, "strip", z.ZodTypeAny, {}, {}>;
}, "strip", z.ZodTypeAny, {
    type: "created_by";
    name: string;
    id: string;
    created_by: {};
    description?: string | undefined;
}, {
    type: "created_by";
    name: string;
    id: string;
    created_by: {};
    description?: string | undefined;
}>, z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    type: z.ZodLiteral<"created_time">;
    created_time: z.ZodObject<{}, "strip", z.ZodTypeAny, {}, {}>;
}, "strip", z.ZodTypeAny, {
    type: "created_time";
    name: string;
    id: string;
    created_time: {};
    description?: string | undefined;
}, {
    type: "created_time";
    name: string;
    id: string;
    created_time: {};
    description?: string | undefined;
}>, z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    type: z.ZodLiteral<"date">;
    date: z.ZodObject<{}, "strip", z.ZodTypeAny, {}, {}>;
}, "strip", z.ZodTypeAny, {
    type: "date";
    date: {};
    name: string;
    id: string;
    description?: string | undefined;
}, {
    type: "date";
    date: {};
    name: string;
    id: string;
    description?: string | undefined;
}>, z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    type: z.ZodLiteral<"email">;
    email: z.ZodObject<{}, "strip", z.ZodTypeAny, {}, {}>;
}, "strip", z.ZodTypeAny, {
    type: "email";
    name: string;
    email: {};
    id: string;
    description?: string | undefined;
}, {
    type: "email";
    name: string;
    email: {};
    id: string;
    description?: string | undefined;
}>, z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    type: z.ZodLiteral<"files">;
    files: z.ZodObject<{}, "strip", z.ZodTypeAny, {}, {}>;
}, "strip", z.ZodTypeAny, {
    type: "files";
    name: string;
    id: string;
    files: {};
    description?: string | undefined;
}, {
    type: "files";
    name: string;
    id: string;
    files: {};
    description?: string | undefined;
}>, z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    type: z.ZodLiteral<"formula">;
    formula: z.ZodObject<{
        expression: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        expression: string;
    }, {
        expression: string;
    }>;
}, "strip", z.ZodTypeAny, {
    type: "formula";
    name: string;
    id: string;
    formula: {
        expression: string;
    };
    description?: string | undefined;
}, {
    type: "formula";
    name: string;
    id: string;
    formula: {
        expression: string;
    };
    description?: string | undefined;
}>, z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    type: z.ZodLiteral<"last_edited_by">;
    last_edited_by: z.ZodObject<{}, "strip", z.ZodTypeAny, {}, {}>;
}, "strip", z.ZodTypeAny, {
    type: "last_edited_by";
    name: string;
    id: string;
    last_edited_by: {};
    description?: string | undefined;
}, {
    type: "last_edited_by";
    name: string;
    id: string;
    last_edited_by: {};
    description?: string | undefined;
}>, z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    type: z.ZodLiteral<"last_edited_time">;
    last_edited_time: z.ZodObject<{}, "strip", z.ZodTypeAny, {}, {}>;
}, "strip", z.ZodTypeAny, {
    type: "last_edited_time";
    name: string;
    id: string;
    last_edited_time: {};
    description?: string | undefined;
}, {
    type: "last_edited_time";
    name: string;
    id: string;
    last_edited_time: {};
    description?: string | undefined;
}>, z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    type: z.ZodLiteral<"multi_select">;
    multi_select: z.ZodObject<{
        options: z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            name: z.ZodString;
            color: z.ZodEnum<["default", "gray", "brown", "orange", "yellow", "green", "blue", "purple", "pink", "red"]>;
        }, "strip", z.ZodTypeAny, {
            name: string;
            color: "default" | "gray" | "brown" | "orange" | "yellow" | "green" | "blue" | "purple" | "pink" | "red";
            id: string;
        }, {
            name: string;
            color: "default" | "gray" | "brown" | "orange" | "yellow" | "green" | "blue" | "purple" | "pink" | "red";
            id: string;
        }>, "many">;
    }, "strip", z.ZodTypeAny, {
        options: {
            name: string;
            color: "default" | "gray" | "brown" | "orange" | "yellow" | "green" | "blue" | "purple" | "pink" | "red";
            id: string;
        }[];
    }, {
        options: {
            name: string;
            color: "default" | "gray" | "brown" | "orange" | "yellow" | "green" | "blue" | "purple" | "pink" | "red";
            id: string;
        }[];
    }>;
}, "strip", z.ZodTypeAny, {
    type: "multi_select";
    name: string;
    id: string;
    multi_select: {
        options: {
            name: string;
            color: "default" | "gray" | "brown" | "orange" | "yellow" | "green" | "blue" | "purple" | "pink" | "red";
            id: string;
        }[];
    };
    description?: string | undefined;
}, {
    type: "multi_select";
    name: string;
    id: string;
    multi_select: {
        options: {
            name: string;
            color: "default" | "gray" | "brown" | "orange" | "yellow" | "green" | "blue" | "purple" | "pink" | "red";
            id: string;
        }[];
    };
    description?: string | undefined;
}>, z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    type: z.ZodLiteral<"number">;
    number: z.ZodObject<{
        format: z.ZodEnum<["number", "number_with_commas", "percent", "dollar", "canadian_dollar", "singapore_dollar", "euro", "pound", "yen", "ruble", "rupee", "won", "yuan", "real", "lira", "rupiah", "franc", "hong_kong_dollar", "new_zealand_dollar", "krona", "norwegian_krone", "mexican_peso", "rand", "new_taiwan_dollar", "danish_krone", "zloty", "baht", "forint", "koruna", "shekel", "chilean_peso", "philippine_peso", "dirham", "colombian_peso", "riyal", "ringgit", "leu", "argentine_peso", "uruguayan_peso", "peruvian_sol"]>;
    }, "strip", z.ZodTypeAny, {
        format: "number" | "percent" | "number_with_commas" | "dollar" | "canadian_dollar" | "singapore_dollar" | "euro" | "pound" | "yen" | "ruble" | "rupee" | "won" | "yuan" | "real" | "lira" | "rupiah" | "franc" | "hong_kong_dollar" | "new_zealand_dollar" | "krona" | "norwegian_krone" | "mexican_peso" | "rand" | "new_taiwan_dollar" | "danish_krone" | "zloty" | "baht" | "forint" | "koruna" | "shekel" | "chilean_peso" | "philippine_peso" | "dirham" | "colombian_peso" | "riyal" | "ringgit" | "leu" | "argentine_peso" | "uruguayan_peso" | "peruvian_sol";
    }, {
        format: "number" | "percent" | "number_with_commas" | "dollar" | "canadian_dollar" | "singapore_dollar" | "euro" | "pound" | "yen" | "ruble" | "rupee" | "won" | "yuan" | "real" | "lira" | "rupiah" | "franc" | "hong_kong_dollar" | "new_zealand_dollar" | "krona" | "norwegian_krone" | "mexican_peso" | "rand" | "new_taiwan_dollar" | "danish_krone" | "zloty" | "baht" | "forint" | "koruna" | "shekel" | "chilean_peso" | "philippine_peso" | "dirham" | "colombian_peso" | "riyal" | "ringgit" | "leu" | "argentine_peso" | "uruguayan_peso" | "peruvian_sol";
    }>;
}, "strip", z.ZodTypeAny, {
    number: {
        format: "number" | "percent" | "number_with_commas" | "dollar" | "canadian_dollar" | "singapore_dollar" | "euro" | "pound" | "yen" | "ruble" | "rupee" | "won" | "yuan" | "real" | "lira" | "rupiah" | "franc" | "hong_kong_dollar" | "new_zealand_dollar" | "krona" | "norwegian_krone" | "mexican_peso" | "rand" | "new_taiwan_dollar" | "danish_krone" | "zloty" | "baht" | "forint" | "koruna" | "shekel" | "chilean_peso" | "philippine_peso" | "dirham" | "colombian_peso" | "riyal" | "ringgit" | "leu" | "argentine_peso" | "uruguayan_peso" | "peruvian_sol";
    };
    type: "number";
    name: string;
    id: string;
    description?: string | undefined;
}, {
    number: {
        format: "number" | "percent" | "number_with_commas" | "dollar" | "canadian_dollar" | "singapore_dollar" | "euro" | "pound" | "yen" | "ruble" | "rupee" | "won" | "yuan" | "real" | "lira" | "rupiah" | "franc" | "hong_kong_dollar" | "new_zealand_dollar" | "krona" | "norwegian_krone" | "mexican_peso" | "rand" | "new_taiwan_dollar" | "danish_krone" | "zloty" | "baht" | "forint" | "koruna" | "shekel" | "chilean_peso" | "philippine_peso" | "dirham" | "colombian_peso" | "riyal" | "ringgit" | "leu" | "argentine_peso" | "uruguayan_peso" | "peruvian_sol";
    };
    type: "number";
    name: string;
    id: string;
    description?: string | undefined;
}>, z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    type: z.ZodLiteral<"people">;
    people: z.ZodObject<{}, "strip", z.ZodTypeAny, {}, {}>;
}, "strip", z.ZodTypeAny, {
    type: "people";
    name: string;
    id: string;
    people: {};
    description?: string | undefined;
}, {
    type: "people";
    name: string;
    id: string;
    people: {};
    description?: string | undefined;
}>, z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    type: z.ZodLiteral<"phone_number">;
    phone_number: z.ZodObject<{}, "strip", z.ZodTypeAny, {}, {}>;
}, "strip", z.ZodTypeAny, {
    type: "phone_number";
    name: string;
    id: string;
    phone_number: {};
    description?: string | undefined;
}, {
    type: "phone_number";
    name: string;
    id: string;
    phone_number: {};
    description?: string | undefined;
}>, z.ZodObject<{
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    type: z.ZodLiteral<"place">;
    place: z.ZodObject<{}, "strip", z.ZodTypeAny, {}, {}>;
}, "strip", z.ZodTypeAny, {
    type: "place";
    name: string;
    place: {};
    description?: string | undefined;
}, {
    type: "place";
    name: string;
    place: {};
    description?: string | undefined;
}>, z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    type: z.ZodLiteral<"relation">;
    relation: z.ZodObject<{
        data_source_id: z.ZodString;
        dual_property: z.ZodOptional<z.ZodObject<{
            synced_property_id: z.ZodString;
            synced_property_name: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            synced_property_id: string;
            synced_property_name: string;
        }, {
            synced_property_id: string;
            synced_property_name: string;
        }>>;
    }, "strip", z.ZodTypeAny, {
        data_source_id: string;
        dual_property?: {
            synced_property_id: string;
            synced_property_name: string;
        } | undefined;
    }, {
        data_source_id: string;
        dual_property?: {
            synced_property_id: string;
            synced_property_name: string;
        } | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    type: "relation";
    name: string;
    id: string;
    relation: {
        data_source_id: string;
        dual_property?: {
            synced_property_id: string;
            synced_property_name: string;
        } | undefined;
    };
    description?: string | undefined;
}, {
    type: "relation";
    name: string;
    id: string;
    relation: {
        data_source_id: string;
        dual_property?: {
            synced_property_id: string;
            synced_property_name: string;
        } | undefined;
    };
    description?: string | undefined;
}>, z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    type: z.ZodLiteral<"rich_text">;
    rich_text: z.ZodObject<{}, "strip", z.ZodTypeAny, {}, {}>;
}, "strip", z.ZodTypeAny, {
    type: "rich_text";
    name: string;
    id: string;
    rich_text: {};
    description?: string | undefined;
}, {
    type: "rich_text";
    name: string;
    id: string;
    rich_text: {};
    description?: string | undefined;
}>, z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    type: z.ZodLiteral<"rollup">;
    rollup: z.ZodObject<{
        function: z.ZodEnum<["average", "checked", "count_per_group", "count", "count_values", "date_range", "earliest_date", "empty", "latest_date", "max", "median", "min", "not_empty", "percent_checked", "percent_empty", "percent_not_empty", "percent_per_group", "percent_unchecked", "range", "unchecked", "unique", "show_original", "show_unique", "sum"]>;
        relation_property_id: z.ZodString;
        relation_property_name: z.ZodString;
        rollup_property_id: z.ZodString;
        rollup_property_name: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        function: "min" | "max" | "count" | "range" | "average" | "checked" | "count_per_group" | "count_values" | "date_range" | "earliest_date" | "empty" | "latest_date" | "median" | "not_empty" | "percent_checked" | "percent_empty" | "percent_not_empty" | "percent_per_group" | "percent_unchecked" | "unchecked" | "unique" | "show_original" | "show_unique" | "sum";
        relation_property_id: string;
        relation_property_name: string;
        rollup_property_id: string;
        rollup_property_name: string;
    }, {
        function: "min" | "max" | "count" | "range" | "average" | "checked" | "count_per_group" | "count_values" | "date_range" | "earliest_date" | "empty" | "latest_date" | "median" | "not_empty" | "percent_checked" | "percent_empty" | "percent_not_empty" | "percent_per_group" | "percent_unchecked" | "unchecked" | "unique" | "show_original" | "show_unique" | "sum";
        relation_property_id: string;
        relation_property_name: string;
        rollup_property_id: string;
        rollup_property_name: string;
    }>;
}, "strip", z.ZodTypeAny, {
    type: "rollup";
    name: string;
    id: string;
    rollup: {
        function: "min" | "max" | "count" | "range" | "average" | "checked" | "count_per_group" | "count_values" | "date_range" | "earliest_date" | "empty" | "latest_date" | "median" | "not_empty" | "percent_checked" | "percent_empty" | "percent_not_empty" | "percent_per_group" | "percent_unchecked" | "unchecked" | "unique" | "show_original" | "show_unique" | "sum";
        relation_property_id: string;
        relation_property_name: string;
        rollup_property_id: string;
        rollup_property_name: string;
    };
    description?: string | undefined;
}, {
    type: "rollup";
    name: string;
    id: string;
    rollup: {
        function: "min" | "max" | "count" | "range" | "average" | "checked" | "count_per_group" | "count_values" | "date_range" | "earliest_date" | "empty" | "latest_date" | "median" | "not_empty" | "percent_checked" | "percent_empty" | "percent_not_empty" | "percent_per_group" | "percent_unchecked" | "unchecked" | "unique" | "show_original" | "show_unique" | "sum";
        relation_property_id: string;
        relation_property_name: string;
        rollup_property_id: string;
        rollup_property_name: string;
    };
    description?: string | undefined;
}>, z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    type: z.ZodLiteral<"select">;
    select: z.ZodObject<{
        options: z.ZodArray<z.ZodObject<{
            id: z.ZodString;
            name: z.ZodString;
            color: z.ZodEnum<["default", "gray", "brown", "orange", "yellow", "green", "blue", "purple", "pink", "red"]>;
        }, "strip", z.ZodTypeAny, {
            name: string;
            color: "default" | "gray" | "brown" | "orange" | "yellow" | "green" | "blue" | "purple" | "pink" | "red";
            id: string;
        }, {
            name: string;
            color: "default" | "gray" | "brown" | "orange" | "yellow" | "green" | "blue" | "purple" | "pink" | "red";
            id: string;
        }>, "many">;
    }, "strip", z.ZodTypeAny, {
        options: {
            name: string;
            color: "default" | "gray" | "brown" | "orange" | "yellow" | "green" | "blue" | "purple" | "pink" | "red";
            id: string;
        }[];
    }, {
        options: {
            name: string;
            color: "default" | "gray" | "brown" | "orange" | "yellow" | "green" | "blue" | "purple" | "pink" | "red";
            id: string;
        }[];
    }>;
}, "strip", z.ZodTypeAny, {
    type: "select";
    name: string;
    id: string;
    select: {
        options: {
            name: string;
            color: "default" | "gray" | "brown" | "orange" | "yellow" | "green" | "blue" | "purple" | "pink" | "red";
            id: string;
        }[];
    };
    description?: string | undefined;
}, {
    type: "select";
    name: string;
    id: string;
    select: {
        options: {
            name: string;
            color: "default" | "gray" | "brown" | "orange" | "yellow" | "green" | "blue" | "purple" | "pink" | "red";
            id: string;
        }[];
    };
    description?: string | undefined;
}>, z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    type: z.ZodLiteral<"title">;
    title: z.ZodObject<{}, "strip", z.ZodTypeAny, {}, {}>;
}, "strip", z.ZodTypeAny, {
    type: "title";
    title: {};
    name: string;
    id: string;
    description?: string | undefined;
}, {
    type: "title";
    title: {};
    name: string;
    id: string;
    description?: string | undefined;
}>, z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    type: z.ZodLiteral<"url">;
    url: z.ZodObject<{}, "strip", z.ZodTypeAny, {}, {}>;
}, "strip", z.ZodTypeAny, {
    type: "url";
    name: string;
    url: {};
    id: string;
    description?: string | undefined;
}, {
    type: "url";
    name: string;
    url: {};
    id: string;
    description?: string | undefined;
}>, z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    type: z.ZodLiteral<"unique_id">;
    unique_id: z.ZodObject<{
        prefix: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        prefix?: string | undefined;
    }, {
        prefix?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    type: "unique_id";
    name: string;
    id: string;
    unique_id: {
        prefix?: string | undefined;
    };
    description?: string | undefined;
}, {
    type: "unique_id";
    name: string;
    id: string;
    unique_id: {
        prefix?: string | undefined;
    };
    description?: string | undefined;
}>]>;
export { ColorEnum, CheckboxPropertySchema, CreatedByPropertySchema, CreatedTimePropertySchema, DatePropertySchema, EmailPropertySchema, FilesPropertySchema, FormulaPropertySchema, LastEditedByPropertySchema, LastEditedTimePropertySchema, MultiSelectPropertySchema, MultiSelectOptionSchema, NumberPropertySchema, NumberFormatEnum, PeoplePropertySchema, PhoneNumberPropertySchema, PlacePropertySchema, RelationPropertySchema, RichTextPropertySchema, RollupPropertySchema, RollupFunctionEnum, SelectPropertySchema, SelectOptionSchema, TitlePropertySchema, URLPropertySchema, UniqueIDPropertySchema, };
//# sourceMappingURL=property-schemas.d.ts.map