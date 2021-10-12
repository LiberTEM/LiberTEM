import * as pathfind from 'path';
import { FormikErrors, FormikValues, withFormik } from 'formik';
import { DatasetTypes } from "../messages";
import { OpenFormProps } from "./types";
import { validateOpen } from "./validate";

export const parseNumList = (nums: string): number[] => nums.split(",").filter(part => part.trim() !== "").map(part => +part)

export const parseNumListWithPadding = (nums: string, minLength: number, maxLength: number): string[] => {
    let initialList = new Array<string>(minLength).fill("");
    if (nums) {
        let value = nums.split(",");
        if (minLength > value.length) {
            initialList = [...value, ...Array<string>(minLength - value.length).fill("")];
        } else {
            value = value.slice(0, maxLength);
            initialList = [...value];
        }
    }
    return initialList;
}

export const parseNumListProduct = (nums: string): number => parseNumList(nums).reduce((a,b) => a * b, 1)

export const frameCalcForOffset = (syncOffset: number, navShapeProduct: number, imageCount: number) => ({
    framesSkippedStart: Math.max(0, syncOffset),
    framesIgnoredEnd: Math.max(0, imageCount - navShapeProduct - syncOffset),
    framesInsertedStart: Math.abs(Math.min(0, syncOffset)),
    framesInsertedEnd: Math.max(0, navShapeProduct - imageCount + syncOffset),
})

export const isSigShapeValid = (sigShape: string, nativeSigShape: string): boolean => parseNumListProduct(sigShape) === parseNumListProduct(nativeSigShape)

export const isSyncOffsetValid = (syncOffset: number, imageCount: number): boolean => -imageCount < syncOffset && syncOffset < imageCount

export const validateSyncOffsetAndSigShape = (
    nativeSigShape: number[] | undefined,
    sigShape: string,
    syncOffset: number,
    imageCount: number | undefined
): FormikErrors<FormikValues> => {
    const res: FormikErrors<FormikValues> = {};
    if (nativeSigShape && !isSigShapeValid(sigShape, nativeSigShape.toString())) {
        res.sig_shape = `must be of size: ${parseNumListProduct(nativeSigShape.toString())}`;
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

export const withValidation = <SubmitParams, FormParams, FormInfo>(
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
