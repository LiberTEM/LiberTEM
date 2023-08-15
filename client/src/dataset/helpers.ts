import * as pathfind from 'path-browserify';
import { FormikErrors, FormikValues, withFormik } from 'formik';
import { DatasetTypes, ShapeLengths } from "../messages";
import { OpenFormProps } from "./types";
import { validateOpen } from "./validate";

export const parseShapeInCommaSeparatedString = (shape: string): number[] => shape.split(",").filter(dim => dim.trim() !== "").map(dim => parseInt(dim, 10))

export const parseShapeInStringArray = (shape: string[]): number[] => shape.filter(dim => dim.trim() !== "").map(dim => parseInt(dim, 10))

export const getMinAndMaxShapeLength = (shapeType: "nav"|"sig"): {minLength: number, maxLength: number} => {
    if (shapeType === "nav") {
        return { minLength: ShapeLengths.NAV_SHAPE_MIN_LENGTH, maxLength: ShapeLengths.NAV_SHAPE_MAX_LENGTH };
    } else {
        return { minLength: ShapeLengths.SIG_SHAPE_MIN_LENGTH, maxLength: ShapeLengths.SIG_SHAPE_MAX_LENGTH };
    }
}

export const adjustShapeWithBounds = (shape: string, shapeType: "nav"|"sig"): string => {
    const { minLength, maxLength } = getMinAndMaxShapeLength(shapeType);

    let adjustedShape = new Array<string>(minLength).fill("");

    if (shape) {
        const parsedShape = shape.split(",");

        if (parsedShape.length === minLength) {
            adjustedShape = [...parsedShape];
        } else if (parsedShape.length < minLength) {
            adjustedShape = [...parsedShape, ...Array<string>(minLength - parsedShape.length).fill("1")];
        } else {
            if (parsedShape.length <= maxLength) {
                adjustedShape = [...parsedShape];
            } else {
                adjustedShape = new Array<string>(maxLength).fill("");
                adjustedShape = [...parsedShape.slice(0, maxLength - 1), productOfShapeInStringArray(parsedShape.slice(maxLength - 1)).toString()];
            }
        }
    }

    return adjustedShape.toString();
}

export const productOfShapeInCommaSeparatedString = (shape: string): number => parseShapeInCommaSeparatedString(shape).reduce((a, b) => a * b, 1)

export const productOfShapeInStringArray = (shape: string[]): number => parseShapeInStringArray(shape).reduce((a, b) => a * b, 1)

export const framesInfoAfterOffsetCorrection = (syncOffset: number, navShapeProduct: number, imageCount: number) => ({
    framesSkippedStart: Math.max(0, syncOffset),
    framesIgnoredEnd: Math.max(0, imageCount - navShapeProduct - syncOffset),
    framesInsertedStart: Math.abs(Math.min(0, syncOffset)),
    framesInsertedEnd: Math.max(0, navShapeProduct - imageCount + syncOffset),
})

export const isSigShapeValid = (sigShape: string, nativeSigShape: string): boolean => productOfShapeInCommaSeparatedString(sigShape) === productOfShapeInCommaSeparatedString(nativeSigShape)

export const isSyncOffsetValid = (syncOffset: number, imageCount: number): boolean => -imageCount < syncOffset && syncOffset < imageCount

export const validateSyncOffsetAndSigShape = (
    nativeSigShape: number[] | undefined,
    sigShape: string,
    syncOffset: number,
    imageCount: number | undefined,
    strictSigShape = false,
): FormikErrors<FormikValues> => {
    const res: FormikErrors<FormikValues> = {};
    if (nativeSigShape && !isSigShapeValid(sigShape, nativeSigShape.toString())) {
        res.sig_shape = `must be of size: ${productOfShapeInCommaSeparatedString(nativeSigShape.toString())}`;
    }
    if (nativeSigShape && strictSigShape && (sigShape !== nativeSigShape.toString())) {
        res.sig_shape = `sig_shape must be equal to: ${nativeSigShape.toString()}`;
    }
    if(imageCount && !isSyncOffsetValid(syncOffset, imageCount)) {
        res.sync_offset = `must be in (-${imageCount}, ${imageCount})`;
    }
    return res;
}

export const getInitial = <T, K extends keyof T, V>(key: K, otherwise: V, values?: T): V | T[K] => {
    if (!values) {
        return otherwise;
    }
    const res = values[key] !== undefined ? values[key] : otherwise;
    return res;
}

export const getInitialName = <T, K extends keyof T>(key: K, otherwise: string, values?: T): string | T[K] => {
    if (!values) {
        return pathfind.basename(otherwise);
    }
    const res = values[key] !== undefined ? values[key] : pathfind.basename(otherwise);
    return res;
}

export const isKnownDatasetType = (detectedType: string): boolean => (Object.keys(DatasetTypes).some((v) => v === detectedType) ? true : false)

type FormToJsonFn<SubmitParams, FormParams> = (inParams: FormParams, path: string) => SubmitParams;
type PropsToValuesFn<SubmitParams, FormParams, FormInfo> = (props: OpenFormProps<SubmitParams, FormInfo>) => FormParams;
type CustomValidationFn<SubmitParams, FormParams, FormInfo> = (inParams: FormParams, props: OpenFormProps<SubmitParams, FormInfo>) => FormikErrors<FormikValues>;

interface WithValidationOpts<SubmitParams, FormParams, FormInfo> {
    formToJson: FormToJsonFn<SubmitParams, FormParams>,
    mapPropsToValues: PropsToValuesFn<SubmitParams, FormParams, FormInfo>,
    type: DatasetTypes,
    customValidation?: CustomValidationFn<SubmitParams, FormParams, FormInfo>
    // WrappedComponent: React.FunctionComponent<FormikProps<FormParams> & OpenFormProps<SubmitParams>>
}

export const withValidation = <SubmitParams, FormParams extends FormikValues, FormInfo>(
    opts: WithValidationOpts<SubmitParams, FormParams, FormInfo>
) => withFormik<OpenFormProps<SubmitParams, FormInfo>, FormParams, FormInfo>({
    mapPropsToValues: opts.mapPropsToValues,
    handleSubmit: (values, formikBag) => {
        const { onSubmit, path } = formikBag.props;
        const submitData = opts.formToJson(values, path);
        onSubmit(submitData);
        formikBag.setSubmitting(false);
    },
    validate: (values, props) => validateOpen(
        props.datasetTypeInfo.schema,
        opts.formToJson(values, props.path),
        opts.customValidation?.(values, props)
    ),
    enableReinitialize: true,
    validateOnChange: true,
    validateOnBlur: true,
})
