import { Text } from "office-ui-fabric-react";
import { Stack, StackItem } from "office-ui-fabric-react/lib/Stack";
import React from "react";
import { IAccuracyPickerProps } from "../FairnessWizard";
import { PredictionTypes } from "../IFairnessProps";
import { IWizardTabProps } from "../IWizardTabProps";
import { localization } from "../Localization/localization";
import { AccuracyTabStyles } from "./AccuracyTab.styles";
import { DataSpecificationBlade } from "./DataSpecificationBlade";
import { TileList } from "./TileList";
import { WizardFooter } from "./WizardFooter";

export interface IAccuracyPickingTabProps extends IWizardTabProps {
    accuracyPickerProps: IAccuracyPickerProps
}

export class AccuracyTab extends React.PureComponent<IAccuracyPickingTabProps> {
    render(): React.ReactNode {
        const styles = AccuracyTabStyles();
        return(
            <Stack horizontal horizontalAlign="space-between" className={styles.frame}>
                <Stack className={styles.main}>
                    {/* <h2 className={styles.header}>
                        {localization.Accuracy.header}
                    </h2> */}
                    <Text className={styles.header}>{localization.Accuracy.header}</Text>
                    {/* <p className={styles.textBody}>{localization.formatString(localization.Accuracy.body,
                        this.props.dashboardContext.modelMetadata.predictionType !== PredictionTypes.regression ?
                            localization.Accuracy.binary :
                            localization.Accuracy.continuous,
                        this.props.dashboardContext.modelMetadata.predictionType === PredictionTypes.binaryClassification ?
                            localization.Accuracy.binary :
                            localization.Accuracy.continuous,
                        this.props.dashboardContext.predictions.length === 1 ?
                            localization.Accuracy.modelMakes :
                            localization.Accuracy.modelsMake
                        )}</p> */}
                        <Text variant={"large"} className={styles.textBody}>{localization.formatString(localization.Accuracy.body,
                        this.props.dashboardContext.modelMetadata.predictionType !== PredictionTypes.regression ?
                            localization.Accuracy.binary :
                            localization.Accuracy.continuous,
                        this.props.dashboardContext.modelMetadata.predictionType === PredictionTypes.binaryClassification ?
                            localization.Accuracy.binary :
                            localization.Accuracy.continuous,
                        this.props.dashboardContext.predictions.length === 1 ?
                            localization.Accuracy.modelMakes :
                            localization.Accuracy.modelsMake
                        )}</Text>
                    <StackItem grow={2} className={styles.itemsList}>
                        <TileList
                            items={this.props.accuracyPickerProps.accuracyOptions.map((accuracy, index) => {
                                return {
                                    title: accuracy.title,
                                    description: accuracy.description,
                                    onSelect: this.props.accuracyPickerProps.onAccuracyChange.bind(this, accuracy.key),
                                    selected: this.props.accuracyPickerProps.selectedAccuracyKey === accuracy.key
                                };
                            })}
                        />
                    </StackItem>
                    <WizardFooter onNext={this.props.onNext} onPrevious={this.props.onPrevious}/>
                </Stack>
                <DataSpecificationBlade
                        numberRows={this.props.dashboardContext.trueY.length}
                        featureNames={this.props.dashboardContext.modelMetadata.featureNames}/>
            </Stack>
        );
    }
}