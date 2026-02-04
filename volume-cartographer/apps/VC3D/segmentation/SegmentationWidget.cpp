#include "SegmentationWidget.hpp"

#include "elements/CollapsibleSettingsGroup.hpp"
#include "VCSettings.hpp"

#include <QAbstractItemView>
#include <QApplication>
#include <QByteArray>
#include <QCheckBox>
#include <QComboBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QEvent>
#include <QGroupBox>
#include <QGridLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QListWidgetItem>
#include <QLoggingCategory>
#include <QMouseEvent>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QRegularExpression>
#include <QScrollBar>
#include <QSettings>
#include <QSignalBlocker>
#include <QSlider>
#include <QSpinBox>
#include <QToolButton>
#include <QVariant>
#include <QVBoxLayout>
#include <QHBoxLayout>

#include <algorithm>
#include <cmath>
#include <exception>

#include <nlohmann/json.hpp>

namespace
{
Q_LOGGING_CATEGORY(lcSegWidget, "vc.segmentation.widget")

constexpr int kGrowDirUpBit = 1 << 0;
constexpr int kGrowDirDownBit = 1 << 1;
constexpr int kGrowDirLeftBit = 1 << 2;
constexpr int kGrowDirRightBit = 1 << 3;
constexpr int kGrowDirAllMask = kGrowDirUpBit | kGrowDirDownBit | kGrowDirLeftBit | kGrowDirRightBit;
constexpr int kCompactDirectionFieldRowLimit = 3;

constexpr float kFloatEpsilon = 1e-4f;
constexpr float kAlphaOpacityScale = 255.0f;

bool nearlyEqual(float lhs, float rhs)
{
    return std::fabs(lhs - rhs) < kFloatEpsilon;
}

float displayOpacityToNormalized(double displayValue)
{
    return static_cast<float>(displayValue / kAlphaOpacityScale);
}

double normalizedOpacityToDisplay(float normalizedValue)
{
    return static_cast<double>(normalizedValue * kAlphaOpacityScale);
}

AlphaPushPullConfig sanitizeAlphaConfig(const AlphaPushPullConfig& config)
{
    AlphaPushPullConfig sanitized = config;

    sanitized.start = std::clamp(sanitized.start, -128.0f, 128.0f);
    sanitized.stop = std::clamp(sanitized.stop, -128.0f, 128.0f);
    if (sanitized.start > sanitized.stop) {
        std::swap(sanitized.start, sanitized.stop);
    }

    const float minStep = 0.05f;
    const float maxStep = 20.0f;
    const float magnitude = std::clamp(std::fabs(sanitized.step), minStep, maxStep);
    sanitized.step = (sanitized.step < 0.0f) ? -magnitude : magnitude;

    sanitized.low = std::clamp(sanitized.low, 0.0f, 1.0f);
    sanitized.high = std::clamp(sanitized.high, 0.0f, 1.0f);
    if (sanitized.high <= sanitized.low + 0.01f) {
        sanitized.high = std::min(1.0f, sanitized.low + 0.05f);
    }

    sanitized.borderOffset = std::clamp(sanitized.borderOffset, -20.0f, 20.0f);
    sanitized.blurRadius = std::clamp(sanitized.blurRadius, 0, 15);
    sanitized.perVertexLimit = std::clamp(sanitized.perVertexLimit, 0.0f, 128.0f);

    return sanitized;
}

bool containsSurfKeyword(const QString& text)
{
    if (text.isEmpty()) {
        return false;
    }
    const QString lowered = text.toLower();
    return lowered.contains(QStringLiteral("surface")) || lowered.contains(QStringLiteral("surf"));
}

std::optional<int> trailingNumber(const QString& text)
{
    static const QRegularExpression numberSuffix(QStringLiteral("(\\d+)$"));
    const auto match = numberSuffix.match(text.trimmed());
    if (match.hasMatch()) {
        return match.captured(1).toInt();
    }
    return std::nullopt;
}

QString settingsGroup()
{
    return QStringLiteral("segmentation_edit");
}
}

QString SegmentationWidget::determineDefaultVolumeId(const QVector<QPair<QString, QString>>& volumes,
                                                     const QString& requestedId) const
{
    const auto hasId = [&volumes](const QString& id) {
        return std::any_of(volumes.cbegin(), volumes.cend(), [&](const auto& entry) {
            return entry.first == id;
        });
    };

    QString numericCandidate;
    int numericValue = -1;
    QString keywordCandidate;

    for (const auto& entry : volumes) {
        const QString& id = entry.first;
        const QString& label = entry.second;

        if (!containsSurfKeyword(id) && !containsSurfKeyword(label)) {
            continue;
        }

        const auto numberFromId = trailingNumber(id);
        const auto numberFromLabel = trailingNumber(label);
        const std::optional<int> number = numberFromId ? numberFromId : numberFromLabel;

        if (number) {
            if (*number > numericValue) {
                numericValue = *number;
                numericCandidate = id;
            }
        } else if (keywordCandidate.isEmpty()) {
            keywordCandidate = id;
        }
    }

    if (!numericCandidate.isEmpty()) {
        return numericCandidate;
    }
    if (!keywordCandidate.isEmpty()) {
        return keywordCandidate;
    }
    if (!requestedId.isEmpty() && hasId(requestedId)) {
        return requestedId;
    }
    if (!volumes.isEmpty()) {
        return volumes.front().first;
    }
    return {};
}

void SegmentationWidget::applyGrowthSteps(int steps, bool persist, bool fromUi)
{
    const int minimum = (_growthMethod == SegmentationGrowthMethod::Corrections) ? 0 : 1;
    const int clamped = std::clamp(steps, minimum, 1024);

    if ((!fromUi || clamped != steps) && _spinGrowthSteps) {
        QSignalBlocker blocker(_spinGrowthSteps);
        _spinGrowthSteps->setValue(clamped);
    }

    if (clamped > 0) {
        _tracerGrowthSteps = std::max(1, clamped);
    }

    _growthSteps = clamped;

    if (persist) {
        writeSetting(QStringLiteral("growth_steps"), _growthSteps);
        writeSetting(QStringLiteral("growth_steps_tracer"), _tracerGrowthSteps);
    }
}

void SegmentationWidget::setGrowthSteps(int steps, bool persist)
{
    applyGrowthSteps(steps, persist, false);
}

SegmentationWidget::SegmentationWidget(QWidget* parent)
    : QWidget(parent)
{
    _growthDirectionMask = kGrowDirAllMask;
    buildUi();
    restoreSettings();
    syncUiState();
}

void SegmentationWidget::buildUi()
{
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(8, 8, 8, 8);
    layout->setSpacing(12);

    auto* editingRow = new QHBoxLayout();
    _chkEditing = new QCheckBox(tr("Enable editing"), this);
    _chkEditing->setToolTip(tr("Start or stop segmentation editing so brush tools can modify surfaces."));
    _lblStatus = new QLabel(this);
    _lblStatus->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    editingRow->addWidget(_chkEditing);
    editingRow->addSpacing(8);
    editingRow->addWidget(_lblStatus, 1);
    layout->addLayout(editingRow);

    _groupGrowth = new QGroupBox(tr("Surface Growth"), this);
    auto* growthLayout = new QVBoxLayout(_groupGrowth);

    // Method selection row
    auto* methodRow = new QHBoxLayout();
    auto* methodLabel = new QLabel(tr("Method:"), _groupGrowth);
    _comboGrowthMethod = new QComboBox(_groupGrowth);
    _comboGrowthMethod->addItem(tr("Tracer"), static_cast<int>(SegmentationGrowthMethod::Tracer));
    _comboGrowthMethod->addItem(tr("Extrapolation"), static_cast<int>(SegmentationGrowthMethod::Extrapolation));
    _comboGrowthMethod->setToolTip(tr("Select the growth algorithm:\n"
                                      "- Tracer: Neural-guided growth using volume data\n"
                                      "- Extrapolation: Simple polynomial extrapolation from boundary points"));
    methodRow->addWidget(methodLabel);
    methodRow->addWidget(_comboGrowthMethod);
    methodRow->addStretch(1);
    growthLayout->addLayout(methodRow);

    // Extrapolation options panel (shown only when Extrapolation method is selected)
    _extrapolationOptionsPanel = new QWidget(_groupGrowth);
    auto* extrapLayout = new QHBoxLayout(_extrapolationOptionsPanel);
    extrapLayout->setContentsMargins(0, 0, 0, 0);
    _lblExtrapolationPoints = new QLabel(tr("Fit points:"), _extrapolationOptionsPanel);
    _spinExtrapolationPoints = new QSpinBox(_extrapolationOptionsPanel);
    _spinExtrapolationPoints->setRange(3, 20);
    _spinExtrapolationPoints->setValue(7);
    _spinExtrapolationPoints->setToolTip(tr("Number of boundary points to use for polynomial fitting."));
    auto* typeLabel = new QLabel(tr("Type:"), _extrapolationOptionsPanel);
    _comboExtrapolationType = new QComboBox(_extrapolationOptionsPanel);
    _comboExtrapolationType->addItem(tr("Linear"), static_cast<int>(ExtrapolationType::Linear));
    _comboExtrapolationType->addItem(tr("Quadratic"), static_cast<int>(ExtrapolationType::Quadratic));
    _comboExtrapolationType->addItem(tr("Linear+Fit"), static_cast<int>(ExtrapolationType::LinearFit));
    _comboExtrapolationType->addItem(tr("Skeleton Path"), static_cast<int>(ExtrapolationType::SkeletonPath));
    _comboExtrapolationType->setToolTip(tr("Extrapolation method:\n"
                                           "- Linear: Fit a straight line (faster, simpler)\n"
                                           "- Quadratic: Fit a parabola (better for curved surfaces)\n"
                                           "- Linear+Fit: Linear extrapolation + Newton refinement to fit selected volume\n"
                                           "- Skeleton Path: Use 2D skeleton analysis + 3D Dijkstra path following"));
    extrapLayout->addWidget(_lblExtrapolationPoints);
    extrapLayout->addWidget(_spinExtrapolationPoints);
    extrapLayout->addSpacing(12);
    extrapLayout->addWidget(typeLabel);
    extrapLayout->addWidget(_comboExtrapolationType);
    extrapLayout->addStretch(1);
    growthLayout->addWidget(_extrapolationOptionsPanel);
    _extrapolationOptionsPanel->setVisible(false);

    // SDT/Newton refinement params (shown only when Linear+Fit is selected)
    _sdtParamsContainer = new QWidget(_groupGrowth);
    auto* sdtLayout = new QHBoxLayout(_sdtParamsContainer);
    sdtLayout->setContentsMargins(0, 0, 0, 0);

    auto* maxStepsLabel = new QLabel(tr("Newton steps:"), _sdtParamsContainer);
    _spinSDTMaxSteps = new QSpinBox(_sdtParamsContainer);
    _spinSDTMaxSteps->setRange(1, 10);
    _spinSDTMaxSteps->setValue(5);
    _spinSDTMaxSteps->setToolTip(tr("Maximum Newton iterations for surface refinement (1-10)."));

    auto* stepSizeLabel = new QLabel(tr("Step size:"), _sdtParamsContainer);
    _spinSDTStepSize = new QDoubleSpinBox(_sdtParamsContainer);
    _spinSDTStepSize->setRange(0.1, 2.0);
    _spinSDTStepSize->setSingleStep(0.1);
    _spinSDTStepSize->setValue(0.8);
    _spinSDTStepSize->setToolTip(tr("Newton step size multiplier (0.1-2.0). Smaller values are more stable."));

    auto* convergenceLabel = new QLabel(tr("Convergence:"), _sdtParamsContainer);
    _spinSDTConvergence = new QDoubleSpinBox(_sdtParamsContainer);
    _spinSDTConvergence->setRange(0.1, 2.0);
    _spinSDTConvergence->setSingleStep(0.1);
    _spinSDTConvergence->setValue(0.5);
    _spinSDTConvergence->setToolTip(tr("Stop refinement when distance < this threshold in voxels (0.1-2.0)."));

    auto* chunkSizeLabel = new QLabel(tr("Chunk:"), _sdtParamsContainer);
    _spinSDTChunkSize = new QSpinBox(_sdtParamsContainer);
    _spinSDTChunkSize->setRange(32, 256);
    _spinSDTChunkSize->setSingleStep(32);
    _spinSDTChunkSize->setValue(128);
    _spinSDTChunkSize->setToolTip(tr("Size of SDT chunks in voxels (32-256). Larger = faster but more memory."));

    sdtLayout->addWidget(maxStepsLabel);
    sdtLayout->addWidget(_spinSDTMaxSteps);
    sdtLayout->addSpacing(8);
    sdtLayout->addWidget(stepSizeLabel);
    sdtLayout->addWidget(_spinSDTStepSize);
    sdtLayout->addSpacing(8);
    sdtLayout->addWidget(convergenceLabel);
    sdtLayout->addWidget(_spinSDTConvergence);
    sdtLayout->addSpacing(8);
    sdtLayout->addWidget(chunkSizeLabel);
    sdtLayout->addWidget(_spinSDTChunkSize);
    sdtLayout->addStretch(1);
    growthLayout->addWidget(_sdtParamsContainer);
    _sdtParamsContainer->setVisible(false);

    // Skeleton path params (shown only when Skeleton Path is selected)
    _skeletonParamsContainer = new QWidget(_groupGrowth);
    auto* skeletonLayout = new QHBoxLayout(_skeletonParamsContainer);
    skeletonLayout->setContentsMargins(0, 0, 0, 0);

    auto* connectivityLabel = new QLabel(tr("Connectivity:"), _skeletonParamsContainer);
    _comboSkeletonConnectivity = new QComboBox(_skeletonParamsContainer);
    _comboSkeletonConnectivity->addItem(tr("6"), 6);
    _comboSkeletonConnectivity->addItem(tr("18"), 18);
    _comboSkeletonConnectivity->addItem(tr("26"), 26);
    _comboSkeletonConnectivity->setCurrentIndex(2);  // Default to 26
    _comboSkeletonConnectivity->setToolTip(tr("3D neighborhood connectivity for Dijkstra pathfinding:\n"
                                              "- 6: Face neighbors only\n"
                                              "- 18: Face + edge neighbors\n"
                                              "- 26: Face + edge + corner neighbors"));

    auto* sliceOrientLabel = new QLabel(tr("Up/Down slice:"), _skeletonParamsContainer);
    _comboSkeletonSliceOrientation = new QComboBox(_skeletonParamsContainer);
    _comboSkeletonSliceOrientation->addItem(tr("YZ (X-slice)"), 0);
    _comboSkeletonSliceOrientation->addItem(tr("XZ (Y-slice)"), 1);
    _comboSkeletonSliceOrientation->setToolTip(tr("For Up/Down growth, which plane to use for 2D skeleton analysis:\n"
                                                   "- YZ (X-slice): Extract slice perpendicular to X axis\n"
                                                   "- XZ (Y-slice): Extract slice perpendicular to Y axis\n"
                                                   "(Left/Right growth always uses XY Z-slices)"));

    auto* skeletonChunkLabel = new QLabel(tr("Chunk:"), _skeletonParamsContainer);
    _spinSkeletonChunkSize = new QSpinBox(_skeletonParamsContainer);
    _spinSkeletonChunkSize->setRange(32, 256);
    _spinSkeletonChunkSize->setSingleStep(32);
    _spinSkeletonChunkSize->setValue(128);
    _spinSkeletonChunkSize->setToolTip(tr("Size of chunks for binary volume loading (32-256). Larger = faster but more memory."));

    auto* searchRadiusLabel = new QLabel(tr("Search:"), _skeletonParamsContainer);
    _spinSkeletonSearchRadius = new QSpinBox(_skeletonParamsContainer);
    _spinSkeletonSearchRadius->setRange(1, 100);
    _spinSkeletonSearchRadius->setSingleStep(1);
    _spinSkeletonSearchRadius->setValue(5);
    _spinSkeletonSearchRadius->setToolTip(tr("When starting point is on background, search this many pixels for nearest component (1-100)."));

    skeletonLayout->addWidget(connectivityLabel);
    skeletonLayout->addWidget(_comboSkeletonConnectivity);
    skeletonLayout->addSpacing(12);
    skeletonLayout->addWidget(sliceOrientLabel);
    skeletonLayout->addWidget(_comboSkeletonSliceOrientation);
    skeletonLayout->addSpacing(12);
    skeletonLayout->addWidget(skeletonChunkLabel);
    skeletonLayout->addWidget(_spinSkeletonChunkSize);
    skeletonLayout->addSpacing(12);
    skeletonLayout->addWidget(searchRadiusLabel);
    skeletonLayout->addWidget(_spinSkeletonSearchRadius);
    skeletonLayout->addStretch(1);
    growthLayout->addWidget(_skeletonParamsContainer);
    _skeletonParamsContainer->setVisible(false);

    auto* dirRow = new QHBoxLayout();
    auto* stepsLabel = new QLabel(tr("Steps:"), _groupGrowth);
    _spinGrowthSteps = new QSpinBox(_groupGrowth);
    _spinGrowthSteps->setRange(0, 1024);
    _spinGrowthSteps->setSingleStep(1);
    _spinGrowthSteps->setToolTip(tr("Number of iterations to run when growing the segmentation."));
    dirRow->addWidget(stepsLabel);
    dirRow->addWidget(_spinGrowthSteps);
    dirRow->addSpacing(16);

    auto* dirLabel = new QLabel(tr("Allowed directions:"), _groupGrowth);
    dirRow->addWidget(dirLabel);
    auto addDirectionCheckbox = [&](const QString& text) {
        auto* box = new QCheckBox(text, _groupGrowth);
        dirRow->addWidget(box);
        return box;
    };
    _chkGrowthDirUp = addDirectionCheckbox(tr("Up"));
    _chkGrowthDirUp->setToolTip(tr("Allow growth steps to move upward along the volume."));
    _chkGrowthDirDown = addDirectionCheckbox(tr("Down"));
    _chkGrowthDirDown->setToolTip(tr("Allow growth steps to move downward along the volume."));
    _chkGrowthDirLeft = addDirectionCheckbox(tr("Left"));
    _chkGrowthDirLeft->setToolTip(tr("Allow growth steps to move left across the volume."));
    _chkGrowthDirRight = addDirectionCheckbox(tr("Right"));
    _chkGrowthDirRight->setToolTip(tr("Allow growth steps to move right across the volume."));
    dirRow->addStretch(1);
    growthLayout->addLayout(dirRow);

    auto* keybindsRow = new QHBoxLayout();
    _chkGrowthKeybindsEnabled = new QCheckBox(tr("Enable growth keybinds (1-6)"), _groupGrowth);
    _chkGrowthKeybindsEnabled->setToolTip(tr("When enabled, keys 1-6 trigger growth in different directions."));
    _chkGrowthKeybindsEnabled->setChecked(_growthKeybindsEnabled);
    keybindsRow->addWidget(_chkGrowthKeybindsEnabled);
    keybindsRow->addStretch(1);
    growthLayout->addLayout(keybindsRow);

    auto* zRow = new QHBoxLayout();
    _chkCorrectionsUseZRange = new QCheckBox(tr("Limit Z range"), _groupGrowth);
    _chkCorrectionsUseZRange->setToolTip(tr("Restrict growth requests to the specified slice range."));
    zRow->addWidget(_chkCorrectionsUseZRange);
    zRow->addSpacing(12);
    auto* zMinLabel = new QLabel(tr("Z min"), _groupGrowth);
    _spinCorrectionsZMin = new QSpinBox(_groupGrowth);
    _spinCorrectionsZMin->setRange(-100000, 100000);
    _spinCorrectionsZMin->setToolTip(tr("Lowest slice index used when Z range limits are enabled."));
    auto* zMaxLabel = new QLabel(tr("Z max"), _groupGrowth);
    _spinCorrectionsZMax = new QSpinBox(_groupGrowth);
    _spinCorrectionsZMax->setRange(-100000, 100000);
    _spinCorrectionsZMax->setToolTip(tr("Highest slice index used when Z range limits are enabled."));
    zRow->addWidget(zMinLabel);
    zRow->addWidget(_spinCorrectionsZMin);
    zRow->addSpacing(8);
    zRow->addWidget(zMaxLabel);
    zRow->addWidget(_spinCorrectionsZMax);
    zRow->addStretch(1);
    growthLayout->addLayout(zRow);

    auto* growButtonsRow = new QHBoxLayout();
    _btnGrow = new QPushButton(tr("Grow"), _groupGrowth);
    _btnGrow->setToolTip(tr("Run surface growth using the configured steps and directions."));
    growButtonsRow->addWidget(_btnGrow);

    _btnInpaint = new QPushButton(tr("Inpaint"), _groupGrowth);
    _btnInpaint->setToolTip(tr("Resume the current surface and run tracer inpainting without additional growth."));
    growButtonsRow->addWidget(_btnInpaint);
    growButtonsRow->addStretch(1);
    growthLayout->addLayout(growButtonsRow);

    auto* volumeRow = new QHBoxLayout();
    auto* volumeLabel = new QLabel(tr("Volume:"), _groupGrowth);
    _comboVolumes = new QComboBox(_groupGrowth);
    _comboVolumes->setEnabled(false);
    _comboVolumes->setToolTip(tr("Select which volume provides source data for segmentation growth."));
    volumeRow->addWidget(volumeLabel);
    volumeRow->addWidget(_comboVolumes, 1);
    growthLayout->addLayout(volumeRow);

    _groupGrowth->setLayout(growthLayout);
    layout->addWidget(_groupGrowth);

    {
        auto* normalGridRow = new QHBoxLayout();
        _lblNormalGrid = new QLabel(this);
        _lblNormalGrid->setTextFormat(Qt::RichText);
        _lblNormalGrid->setToolTip(tr("Shows whether precomputed normal grids are available for push/pull tools."));
        _lblNormalGrid->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
        normalGridRow->addWidget(_lblNormalGrid, 0);

        _editNormalGridPath = new QLineEdit(this);
        _editNormalGridPath->setReadOnly(true);
        _editNormalGridPath->setClearButtonEnabled(false);
        _editNormalGridPath->setVisible(false);
        normalGridRow->addWidget(_editNormalGridPath, 1);
        layout->addLayout(normalGridRow);
    }

    // Normal3D zarr selection (optional)
    {
        auto* normal3dRow = new QHBoxLayout();
        _lblNormal3d = new QLabel(this);
        _lblNormal3d->setTextFormat(Qt::RichText);
        _lblNormal3d->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
        normal3dRow->addWidget(_lblNormal3d, 0);

        _editNormal3dPath = new QLineEdit(this);
        _editNormal3dPath->setReadOnly(true);
        _editNormal3dPath->setClearButtonEnabled(false);
        _editNormal3dPath->setVisible(false);
        normal3dRow->addWidget(_editNormal3dPath, 1);

        _comboNormal3d = new QComboBox(this);
        _comboNormal3d->setToolTip(tr("Select Normal3D zarr volume to use for normal3dline constraints."));
        _comboNormal3d->setVisible(false);
        normal3dRow->addWidget(_comboNormal3d, 0);

        layout->addLayout(normal3dRow);
    }

    auto* hoverRow = new QHBoxLayout();
    hoverRow->addSpacing(4);
    _chkShowHoverMarker = new QCheckBox(tr("Show hover marker"), this);
    _chkShowHoverMarker->setToolTip(tr("Toggle the hover indicator in the segmentation viewer. "
                                       "Disabling this hides the preview marker and defers grid lookups "
                                       "until you drag or use push/pull."));
    hoverRow->addWidget(_chkShowHoverMarker);
    hoverRow->addStretch(1);
    layout->addLayout(hoverRow);

    _groupEditing = new CollapsibleSettingsGroup(tr("Editing"), this);
    auto* falloffLayout = _groupEditing->contentLayout();
    auto* falloffParent = _groupEditing->contentWidget();

    auto createToolGroup = [&](const QString& title,
                               QDoubleSpinBox*& radiusSpin,
                               QDoubleSpinBox*& sigmaSpin) {
        auto* group = new CollapsibleSettingsGroup(title, _groupEditing);
        radiusSpin = group->addDoubleSpinBox(tr("Radius"), 0.25, 128.0, 0.25);
        sigmaSpin = group->addDoubleSpinBox(tr("Sigma"), 0.05, 64.0, 0.1);
        return group;
    };

    _groupDrag = createToolGroup(tr("Drag Brush"), _spinDragRadius, _spinDragSigma);
    _groupLine = createToolGroup(tr("Line Brush (S)"), _spinLineRadius, _spinLineSigma);

    _groupPushPull = new CollapsibleSettingsGroup(tr("Push/Pull (A / D, Ctrl for alpha)"), _groupEditing);
    auto* pushGrid = new QGridLayout();
    pushGrid->setContentsMargins(0, 0, 0, 0);
    pushGrid->setHorizontalSpacing(12);
    pushGrid->setVerticalSpacing(8);
    _groupPushPull->contentLayout()->addLayout(pushGrid);

    auto* pushParent = _groupPushPull->contentWidget();

    auto* ppRadiusLabel = new QLabel(tr("Radius"), pushParent);
    _spinPushPullRadius = new QDoubleSpinBox(pushParent);
    _spinPushPullRadius->setDecimals(2);
    _spinPushPullRadius->setRange(0.25, 128.0);
    _spinPushPullRadius->setSingleStep(0.25);
    pushGrid->addWidget(ppRadiusLabel, 0, 0);
    pushGrid->addWidget(_spinPushPullRadius, 0, 1);

    auto* ppSigmaLabel = new QLabel(tr("Sigma"), pushParent);
    _spinPushPullSigma = new QDoubleSpinBox(pushParent);
    _spinPushPullSigma->setDecimals(2);
    _spinPushPullSigma->setRange(0.05, 64.0);
    _spinPushPullSigma->setSingleStep(0.1);
    pushGrid->addWidget(ppSigmaLabel, 0, 2);
    pushGrid->addWidget(_spinPushPullSigma, 0, 3);

    auto* pushPullLabel = new QLabel(tr("Step"), pushParent);
    _spinPushPullStep = new QDoubleSpinBox(pushParent);
    _spinPushPullStep->setDecimals(2);
    _spinPushPullStep->setRange(0.05, 40.0);
    _spinPushPullStep->setSingleStep(0.05);
    pushGrid->addWidget(pushPullLabel, 1, 0);
    pushGrid->addWidget(_spinPushPullStep, 1, 1);

    _lblAlphaInfo = new QLabel(tr("Hold Ctrl with A/D to sample alpha while pushing or pulling."), pushParent);
    _lblAlphaInfo->setWordWrap(true);
    _lblAlphaInfo->setToolTip(tr("Hold Ctrl when starting push/pull to stop at the configured alpha thresholds."));
    pushGrid->addWidget(_lblAlphaInfo, 2, 0, 1, 4);

    _alphaPushPullPanel = new QWidget(pushParent);
    auto* alphaGrid = new QGridLayout(_alphaPushPullPanel);
    alphaGrid->setContentsMargins(0, 0, 0, 0);
    alphaGrid->setHorizontalSpacing(12);
    alphaGrid->setVerticalSpacing(6);

    auto addAlphaWidget = [&](const QString& labelText, QWidget* widget, int row, int column, const QString& tooltip) {
        auto* label = new QLabel(labelText, _alphaPushPullPanel);
        label->setToolTip(tooltip);
        widget->setToolTip(tooltip);
        const int columnBase = column * 2;
        alphaGrid->addWidget(label, row, columnBase);
        alphaGrid->addWidget(widget, row, columnBase + 1);
    };

    auto addAlphaControl = [&](const QString& labelText,
                               QDoubleSpinBox*& target,
                               double min,
                               double max,
                               double step,
                               int row,
                               int column,
                               const QString& tooltip) {
        auto* spin = new QDoubleSpinBox(_alphaPushPullPanel);
        spin->setDecimals(2);
        spin->setRange(min, max);
        spin->setSingleStep(step);
        target = spin;
        addAlphaWidget(labelText, spin, row, column, tooltip);
    };

    auto addAlphaIntControl = [&](const QString& labelText,
                                  QSpinBox*& target,
                                  int min,
                                  int max,
                                  int step,
                                  int row,
                                  int column,
                                  const QString& tooltip) {
        auto* spin = new QSpinBox(_alphaPushPullPanel);
        spin->setRange(min, max);
        spin->setSingleStep(step);
        target = spin;
        addAlphaWidget(labelText, spin, row, column, tooltip);
    };

    int alphaRow = 0;
    addAlphaControl(tr("Start"), _spinAlphaStart, -64.0, 64.0, 0.5, alphaRow, 0,
                    tr("Beginning distance (along the brush normal) where alpha sampling starts."));
    addAlphaControl(tr("Stop"), _spinAlphaStop, -64.0, 64.0, 0.5, alphaRow++, 1,
                    tr("Ending distance for alpha sampling; the search stops once this depth is reached."));
    addAlphaControl(tr("Sample step"), _spinAlphaStep, 0.05, 20.0, 0.05, alphaRow, 0,
                    tr("Spacing between alpha samples inside the start/stop range; smaller steps follow fine features."));
    addAlphaControl(tr("Border offset"), _spinAlphaBorder, -20.0, 20.0, 0.1, alphaRow++, 1,
                    tr("Extra offset applied after the alpha front is located, keeping a safety margin."));
    addAlphaControl(tr("Opacity low"), _spinAlphaLow, 0.0, 255.0, 1.0, alphaRow, 0,
                    tr("Lower bound of the opacity window; voxels below this behave as transparent."));
    addAlphaControl(tr("Opacity high"), _spinAlphaHigh, 0.0, 255.0, 1.0, alphaRow++, 1,
                    tr("Upper bound of the opacity window; voxels above this are fully opaque."));

    const QString blurTooltip = tr("Gaussian blur radius for each sampled slice; higher values smooth noisy volumes before thresholding.");
    addAlphaIntControl(tr("Blur radius"), _spinAlphaBlurRadius, 0, 15, 1, alphaRow++, 0, blurTooltip);

    _chkAlphaPerVertex = new QCheckBox(tr("Independent per-vertex stops"), _alphaPushPullPanel);
    _chkAlphaPerVertex->setToolTip(tr("Move every vertex within the brush independently to the alpha threshold without Gaussian weighting."));
    alphaGrid->addWidget(_chkAlphaPerVertex, alphaRow++, 0, 1, 4);

    const QString perVertexLimitTip = tr("Maximum additional distance (world units) a vertex may exceed relative to the smallest movement in the brush when independent stops are enabled.");
    addAlphaControl(tr("Per-vertex limit"), _spinAlphaPerVertexLimit, 0.0, 128.0, 0.25, alphaRow++, 0, perVertexLimitTip);

    alphaGrid->setColumnStretch(1, 1);
    alphaGrid->setColumnStretch(3, 1);

    pushGrid->addWidget(_alphaPushPullPanel, 3, 0, 1, 4);

    pushGrid->setColumnStretch(1, 1);
    pushGrid->setColumnStretch(3, 1);

    auto setGroupTooltips = [](QWidget* group, QDoubleSpinBox* radiusSpin, QDoubleSpinBox* sigmaSpin, const QString& radiusTip, const QString& sigmaTip) {
        if (group) {
            group->setToolTip(radiusTip + QLatin1Char('\n') + sigmaTip);
        }
        if (radiusSpin) {
            radiusSpin->setToolTip(radiusTip);
        }
        if (sigmaSpin) {
            sigmaSpin->setToolTip(sigmaTip);
        }
    };

    setGroupTooltips(_groupDrag,
                     _spinDragRadius,
                     _spinDragSigma,
                     tr("Brush radius in grid steps for drag edits."),
                     tr("Gaussian falloff sigma for drag edits."));
    setGroupTooltips(_groupLine,
                     _spinLineRadius,
                     _spinLineSigma,
                     tr("Brush radius in grid steps for line drags."),
                     tr("Gaussian falloff sigma for line drags."));
    setGroupTooltips(_groupPushPull,
                     _spinPushPullRadius,
                     _spinPushPullSigma,
                     tr("Radius in grid steps that participates in push/pull."),
                     tr("Gaussian falloff sigma for push/pull."));
    if (_spinPushPullStep) {
        _spinPushPullStep->setToolTip(tr("Baseline step size (in world units) for classic push/pull when alpha mode is disabled."));
    }

    auto* brushToolsRow = new QHBoxLayout();
    brushToolsRow->setSpacing(12);
    brushToolsRow->addWidget(_groupDrag, 1);
    brushToolsRow->addWidget(_groupLine, 1);
    falloffLayout->addLayout(brushToolsRow);

    auto* pushPullRow = new QHBoxLayout();
    pushPullRow->setSpacing(12);
    pushPullRow->addWidget(_groupPushPull, 1);
    falloffLayout->addLayout(pushPullRow);

    auto* smoothingRow = new QHBoxLayout();
    auto* smoothStrengthLabel = new QLabel(tr("Smoothing strength"), falloffParent);
    _spinSmoothStrength = new QDoubleSpinBox(falloffParent);
    _spinSmoothStrength->setDecimals(2);
    _spinSmoothStrength->setToolTip(tr("Blend edits toward neighboring vertices; higher values smooth more."));
    _spinSmoothStrength->setRange(0.0, 1.0);
    _spinSmoothStrength->setSingleStep(0.05);
    smoothingRow->addWidget(smoothStrengthLabel);
    smoothingRow->addWidget(_spinSmoothStrength);
    smoothingRow->addSpacing(12);
    auto* smoothIterationsLabel = new QLabel(tr("Iterations"), falloffParent);
    _spinSmoothIterations = new QSpinBox(falloffParent);
    _spinSmoothIterations->setRange(1, 25);
    _spinSmoothIterations->setToolTip(tr("Number of smoothing passes applied after growth."));
    _spinSmoothIterations->setSingleStep(1);
    smoothingRow->addWidget(smoothIterationsLabel);
    smoothingRow->addWidget(_spinSmoothIterations);
    smoothingRow->addStretch(1);
    falloffLayout->addLayout(smoothingRow);

    layout->addWidget(_groupEditing);

    // Approval Mask Group
    _groupApprovalMask = new CollapsibleSettingsGroup(tr("Approval Mask"), this);
    auto* approvalLayout = _groupApprovalMask->contentLayout();
    auto* approvalParent = _groupApprovalMask->contentWidget();

    // Show approval mask checkbox
    _chkShowApprovalMask = new QCheckBox(tr("Show Approval Mask"), approvalParent);
    _chkShowApprovalMask->setToolTip(tr("Display the approval mask overlay on the surface."));
    approvalLayout->addWidget(_chkShowApprovalMask);

    // Edit checkboxes row - mutually exclusive approve/unapprove modes
    auto* editRow = new QHBoxLayout();
    editRow->setSpacing(8);

    _chkEditApprovedMask = new QCheckBox(tr("Edit Approved (B)"), approvalParent);
    _chkEditApprovedMask->setToolTip(tr("Paint regions as approved. Saves to disk when toggled off."));
    _chkEditApprovedMask->setEnabled(false);  // Only enabled when show is checked

    _chkEditUnapprovedMask = new QCheckBox(tr("Edit Unapproved (N)"), approvalParent);
    _chkEditUnapprovedMask->setToolTip(tr("Paint regions as unapproved. Saves to disk when toggled off."));
    _chkEditUnapprovedMask->setEnabled(false);  // Only enabled when show is checked

    editRow->addWidget(_chkEditApprovedMask);
    editRow->addWidget(_chkEditUnapprovedMask);
    editRow->addStretch(1);
    approvalLayout->addLayout(editRow);

    // Cylinder brush controls: radius and depth
    // Radius = circle in plane views, width of rectangle in flattened view
    // Depth = height of rectangle in flattened view, cylinder thickness for plane painting
    auto* approvalBrushRow = new QHBoxLayout();
    approvalBrushRow->setSpacing(8);

    auto* brushRadiusLabel = new QLabel(tr("Radius:"), approvalParent);
    _spinApprovalBrushRadius = new QDoubleSpinBox(approvalParent);
    _spinApprovalBrushRadius->setDecimals(0);
    _spinApprovalBrushRadius->setRange(1.0, 1000.0);
    _spinApprovalBrushRadius->setSingleStep(10.0);
    _spinApprovalBrushRadius->setValue(_approvalBrushRadius);
    _spinApprovalBrushRadius->setToolTip(tr("Cylinder radius: circle size in plane views, rectangle width in flattened view (native voxels)."));
    approvalBrushRow->addWidget(brushRadiusLabel);
    approvalBrushRow->addWidget(_spinApprovalBrushRadius);

    auto* brushDepthLabel = new QLabel(tr("Depth:"), approvalParent);
    _spinApprovalBrushDepth = new QDoubleSpinBox(approvalParent);
    _spinApprovalBrushDepth->setDecimals(0);
    _spinApprovalBrushDepth->setRange(1.0, 500.0);
    _spinApprovalBrushDepth->setSingleStep(5.0);
    _spinApprovalBrushDepth->setValue(_approvalBrushDepth);
    _spinApprovalBrushDepth->setToolTip(tr("Cylinder depth: rectangle height in flattened view, painting thickness from plane views (native voxels)."));
    approvalBrushRow->addWidget(brushDepthLabel);
    approvalBrushRow->addWidget(_spinApprovalBrushDepth);
    approvalBrushRow->addStretch(1);
    approvalLayout->addLayout(approvalBrushRow);

    // Opacity slider row
    auto* opacityRow = new QHBoxLayout();
    opacityRow->setSpacing(8);

    auto* opacityLabel = new QLabel(tr("Opacity:"), approvalParent);
    _sliderApprovalMaskOpacity = new QSlider(Qt::Horizontal, approvalParent);
    _sliderApprovalMaskOpacity->setRange(0, 100);
    _sliderApprovalMaskOpacity->setValue(_approvalMaskOpacity);
    _sliderApprovalMaskOpacity->setToolTip(tr("Mask overlay transparency (0 = transparent, 100 = opaque)."));

    _lblApprovalMaskOpacity = new QLabel(QString::number(_approvalMaskOpacity) + QStringLiteral("%"), approvalParent);
    _lblApprovalMaskOpacity->setMinimumWidth(35);

    opacityRow->addWidget(opacityLabel);
    opacityRow->addWidget(_sliderApprovalMaskOpacity, 1);
    opacityRow->addWidget(_lblApprovalMaskOpacity);
    approvalLayout->addLayout(opacityRow);

    // Color picker row
    auto* colorRow = new QHBoxLayout();
    colorRow->setSpacing(8);

    auto* colorLabel = new QLabel(tr("Brush Color:"), approvalParent);
    _btnApprovalColor = new QPushButton(approvalParent);
    _btnApprovalColor->setFixedSize(60, 24);
    _btnApprovalColor->setToolTip(tr("Click to choose the color for approval mask painting."));
    // Set initial color preview
    _btnApprovalColor->setStyleSheet(
        QStringLiteral("background-color: %1; border: 1px solid #888;").arg(_approvalBrushColor.name()));

    colorRow->addWidget(colorLabel);
    colorRow->addWidget(_btnApprovalColor);
    colorRow->addStretch(1);
    approvalLayout->addLayout(colorRow);

    // Undo button
    auto* buttonRow = new QHBoxLayout();
    buttonRow->setSpacing(8);
    _btnUndoApprovalStroke = new QPushButton(tr("Undo (Ctrl+B)"), approvalParent);
    _btnUndoApprovalStroke->setToolTip(tr("Undo the last approval mask brush stroke."));
    buttonRow->addWidget(_btnUndoApprovalStroke);
    buttonRow->addStretch(1);
    approvalLayout->addLayout(buttonRow);

    layout->addWidget(_groupApprovalMask);

    // Cell Reoptimization Group
    _groupCellReopt = new CollapsibleSettingsGroup(tr("Cell Reoptimization"), this);
    auto* cellReoptLayout = _groupCellReopt->contentLayout();
    auto* cellReoptParent = _groupCellReopt->contentWidget();

    // Enable mode checkbox
    _chkCellReoptMode = new QCheckBox(tr("Enable Cell Reoptimization"), cellReoptParent);
    _chkCellReoptMode->setToolTip(tr("Click on unapproved regions to flood fill and place correction points.\n"
                                      "Requires approval mask to be visible."));
    cellReoptLayout->addWidget(_chkCellReoptMode);

    // Max flood cells
    auto* maxFloodRow = new QHBoxLayout();
    maxFloodRow->setSpacing(8);
    auto* maxFloodLabel = new QLabel(tr("Max Flood Cells:"), cellReoptParent);
    _spinCellReoptMaxSteps = new QSpinBox(cellReoptParent);
    _spinCellReoptMaxSteps->setRange(10, 10000);
    _spinCellReoptMaxSteps->setValue(_cellReoptMaxSteps);
    _spinCellReoptMaxSteps->setToolTip(tr("Maximum number of cells to include in the flood fill."));
    maxFloodRow->addWidget(maxFloodLabel);
    maxFloodRow->addWidget(_spinCellReoptMaxSteps);
    maxFloodRow->addStretch(1);
    cellReoptLayout->addLayout(maxFloodRow);

    // Max correction points
    auto* maxPointsRow = new QHBoxLayout();
    maxPointsRow->setSpacing(8);
    auto* maxPointsLabel = new QLabel(tr("Max Points:"), cellReoptParent);
    _spinCellReoptMaxPoints = new QSpinBox(cellReoptParent);
    _spinCellReoptMaxPoints->setRange(3, 200);
    _spinCellReoptMaxPoints->setValue(_cellReoptMaxPoints);
    _spinCellReoptMaxPoints->setToolTip(tr("Maximum number of correction points to place on the boundary."));
    maxPointsRow->addWidget(maxPointsLabel);
    maxPointsRow->addWidget(_spinCellReoptMaxPoints);
    maxPointsRow->addStretch(1);
    cellReoptLayout->addLayout(maxPointsRow);

    // Min point spacing
    auto* minSpacingRow = new QHBoxLayout();
    minSpacingRow->setSpacing(8);
    auto* minSpacingLabel = new QLabel(tr("Min Spacing:"), cellReoptParent);
    _spinCellReoptMinSpacing = new QDoubleSpinBox(cellReoptParent);
    _spinCellReoptMinSpacing->setRange(1.0, 50.0);
    _spinCellReoptMinSpacing->setValue(_cellReoptMinSpacing);
    _spinCellReoptMinSpacing->setSuffix(tr(" grid"));
    _spinCellReoptMinSpacing->setToolTip(tr("Minimum spacing between correction points (grid steps)."));
    minSpacingRow->addWidget(minSpacingLabel);
    minSpacingRow->addWidget(_spinCellReoptMinSpacing);
    minSpacingRow->addStretch(1);
    cellReoptLayout->addLayout(minSpacingRow);

    // Perimeter offset
    auto* perimeterOffsetRow = new QHBoxLayout();
    perimeterOffsetRow->setSpacing(8);
    auto* perimeterOffsetLabel = new QLabel(tr("Perimeter Offset:"), cellReoptParent);
    _spinCellReoptPerimeterOffset = new QDoubleSpinBox(cellReoptParent);
    _spinCellReoptPerimeterOffset->setRange(-50.0, 50.0);
    _spinCellReoptPerimeterOffset->setValue(_cellReoptPerimeterOffset);
    _spinCellReoptPerimeterOffset->setSuffix(tr(" grid"));
    _spinCellReoptPerimeterOffset->setToolTip(tr("Offset to expand (+) or shrink (-) the traced perimeter from center of mass."));
    perimeterOffsetRow->addWidget(perimeterOffsetLabel);
    perimeterOffsetRow->addWidget(_spinCellReoptPerimeterOffset);
    perimeterOffsetRow->addStretch(1);
    cellReoptLayout->addLayout(perimeterOffsetRow);

    // Collection selector
    auto* collectionRow = new QHBoxLayout();
    collectionRow->setSpacing(8);
    auto* collectionLabel = new QLabel(tr("Collection:"), cellReoptParent);
    _comboCellReoptCollection = new QComboBox(cellReoptParent);
    _comboCellReoptCollection->setToolTip(tr("Select which correction point collection to use for reoptimization."));
    _comboCellReoptCollection->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    collectionRow->addWidget(collectionLabel);
    collectionRow->addWidget(_comboCellReoptCollection, 1);
    cellReoptLayout->addLayout(collectionRow);

    // Run reoptimization button
    auto* runButtonRow = new QHBoxLayout();
    runButtonRow->setSpacing(8);
    _btnCellReoptRun = new QPushButton(tr("Run Reoptimization"), cellReoptParent);
    _btnCellReoptRun->setToolTip(tr("Trigger reoptimization using the selected correction point collection."));
    runButtonRow->addWidget(_btnCellReoptRun);
    runButtonRow->addStretch(1);
    cellReoptLayout->addLayout(runButtonRow);

    layout->addWidget(_groupCellReopt);

    _groupDirectionField = new CollapsibleSettingsGroup(tr("Direction Fields"), this);

    auto* directionParent = _groupDirectionField->contentWidget();

    _groupDirectionField->addRow(tr("Zarr folder:"), [&](QHBoxLayout* row) {
        _directionFieldPathEdit = new QLineEdit(directionParent);
        _directionFieldPathEdit->setToolTip(tr("Filesystem path to the direction field zarr folder."));
        _directionFieldBrowseButton = new QToolButton(directionParent);
        _directionFieldBrowseButton->setText(QStringLiteral("..."));
        _directionFieldBrowseButton->setToolTip(tr("Browse for a direction field dataset on disk."));
        row->addWidget(_directionFieldPathEdit, 1);
        row->addWidget(_directionFieldBrowseButton);
    }, tr("Filesystem path to the direction field zarr folder."));

    _groupDirectionField->addRow(tr("Orientation:"), [&](QHBoxLayout* row) {
        _comboDirectionFieldOrientation = new QComboBox(directionParent);
        _comboDirectionFieldOrientation->setToolTip(tr("Select which axis the direction field describes."));
        _comboDirectionFieldOrientation->addItem(tr("Normal"), static_cast<int>(SegmentationDirectionFieldOrientation::Normal));
        _comboDirectionFieldOrientation->addItem(tr("Horizontal"), static_cast<int>(SegmentationDirectionFieldOrientation::Horizontal));
        _comboDirectionFieldOrientation->addItem(tr("Vertical"), static_cast<int>(SegmentationDirectionFieldOrientation::Vertical));
        row->addWidget(_comboDirectionFieldOrientation);
        row->addSpacing(12);

        auto* scaleLabel = new QLabel(tr("Scale level:"), directionParent);
        _comboDirectionFieldScale = new QComboBox(directionParent);
        _comboDirectionFieldScale->setToolTip(tr("Choose the multiscale level sampled from the direction field."));
        for (int scale = 0; scale <= 5; ++scale) {
            _comboDirectionFieldScale->addItem(QString::number(scale), scale);
        }
        row->addWidget(scaleLabel);
        row->addWidget(_comboDirectionFieldScale);
        row->addSpacing(12);

        auto* weightLabel = new QLabel(tr("Weight:"), directionParent);
        _spinDirectionFieldWeight = new QDoubleSpinBox(directionParent);
        _spinDirectionFieldWeight->setDecimals(2);
        _spinDirectionFieldWeight->setToolTip(tr("Relative influence of this direction field during growth."));
        _spinDirectionFieldWeight->setRange(0.0, 10.0);
        _spinDirectionFieldWeight->setSingleStep(0.1);
        row->addWidget(weightLabel);
        row->addWidget(_spinDirectionFieldWeight);
        row->addStretch(1);
    });

    _groupDirectionField->addRow(QString(), [&](QHBoxLayout* row) {
        _directionFieldAddButton = new QPushButton(tr("Add"), directionParent);
        _directionFieldAddButton->setToolTip(tr("Save the current direction field parameters to the list."));
        _directionFieldRemoveButton = new QPushButton(tr("Remove"), directionParent);
        _directionFieldRemoveButton->setToolTip(tr("Delete the selected direction field entry."));
        _directionFieldRemoveButton->setEnabled(false);
        _directionFieldClearButton = new QPushButton(tr("Clear"), directionParent);
        _directionFieldClearButton->setToolTip(tr("Clear selection and reset the form for adding a new entry."));
        row->addWidget(_directionFieldAddButton);
        row->addWidget(_directionFieldRemoveButton);
        row->addWidget(_directionFieldClearButton);
        row->addStretch(1);
    });

    _directionFieldList = new QListWidget(directionParent);
    _directionFieldList->setToolTip(tr("Direction field configurations applied during growth."));
    _directionFieldList->setSelectionMode(QAbstractItemView::SingleSelection);
    _groupDirectionField->addFullWidthWidget(_directionFieldList);

    layout->addWidget(_groupDirectionField);

    // Neural Tracer group
    _groupNeuralTracer = new CollapsibleSettingsGroup(tr("Neural Tracer"), this);
    auto* neuralParent = _groupNeuralTracer->contentWidget();

    _chkNeuralTracerEnabled = new QCheckBox(tr("Enable neural tracer"), neuralParent);
    _chkNeuralTracerEnabled->setToolTip(tr("Use neural network-based tracing instead of the default tracer. "
                                           "Requires a trained model checkpoint."));
    _groupNeuralTracer->contentLayout()->addWidget(_chkNeuralTracerEnabled);

    _groupNeuralTracer->addRow(tr("Checkpoint:"), [&](QHBoxLayout* row) {
        _neuralCheckpointEdit = new QLineEdit(neuralParent);
        _neuralCheckpointEdit->setPlaceholderText(tr("Path to model checkpoint (.pt)"));
        _neuralCheckpointEdit->setToolTip(tr("Path to the trained neural network checkpoint file."));
        _neuralCheckpointBrowse = new QToolButton(neuralParent);
        _neuralCheckpointBrowse->setText(QStringLiteral("..."));
        _neuralCheckpointBrowse->setToolTip(tr("Browse for checkpoint file."));
        row->addWidget(_neuralCheckpointEdit, 1);
        row->addWidget(_neuralCheckpointBrowse);
    }, tr("Path to the trained neural network checkpoint file."));

    _groupNeuralTracer->addRow(tr("Python:"), [&](QHBoxLayout* row) {
        _neuralPythonEdit = new QLineEdit(neuralParent);
        _neuralPythonEdit->setPlaceholderText(tr("Path to Python executable (leave empty for auto-detect)"));
        _neuralPythonEdit->setToolTip(tr("Path to the Python executable with torch installed (e.g. ~/miniconda3/bin/python). "
                                         "Leave empty to auto-detect."));
        _neuralPythonBrowse = new QToolButton(neuralParent);
        _neuralPythonBrowse->setText(QStringLiteral("..."));
        _neuralPythonBrowse->setToolTip(tr("Browse for Python executable."));
        row->addWidget(_neuralPythonEdit, 1);
        row->addWidget(_neuralPythonBrowse);
    }, tr("Python executable with torch installed."));

    _groupNeuralTracer->addRow(tr("Volume scale:"), [&](QHBoxLayout* row) {
        _comboNeuralVolumeScale = new QComboBox(neuralParent);
        _comboNeuralVolumeScale->setToolTip(tr("OME-Zarr scale level to use for neural tracing (0 = full resolution)."));
        for (int scale = 0; scale <= 5; ++scale) {
            _comboNeuralVolumeScale->addItem(QString::number(scale), scale);
        }
        row->addWidget(_comboNeuralVolumeScale);

        auto* batchLabel = new QLabel(tr("Batch size:"), neuralParent);
        _spinNeuralBatchSize = new QSpinBox(neuralParent);
        _spinNeuralBatchSize->setRange(1, 64);
        _spinNeuralBatchSize->setValue(_neuralBatchSize);
        _spinNeuralBatchSize->setToolTip(tr("Number of points to process in parallel (higher = faster but more memory)."));
        row->addSpacing(12);
        row->addWidget(batchLabel);
        row->addWidget(_spinNeuralBatchSize);
        row->addStretch(1);
    });

    _lblNeuralTracerStatus = new QLabel(neuralParent);
    _lblNeuralTracerStatus->setWordWrap(true);
    _lblNeuralTracerStatus->setVisible(false);
    _groupNeuralTracer->contentLayout()->addWidget(_lblNeuralTracerStatus);

    layout->addWidget(_groupNeuralTracer);

    auto rememberGroupState = [this](CollapsibleSettingsGroup* group, const QString& key) {
        if (!group) {
            return;
        }
        connect(group, &CollapsibleSettingsGroup::toggled, this, [this, key](bool expanded) {
            if (_restoringSettings) {
                return;
            }
            writeSetting(key, expanded);
        });
    };

    rememberGroupState(_groupEditing, QStringLiteral("group_editing_expanded"));
    rememberGroupState(_groupDrag, QStringLiteral("group_drag_expanded"));
    rememberGroupState(_groupLine, QStringLiteral("group_line_expanded"));
    rememberGroupState(_groupPushPull, QStringLiteral("group_push_pull_expanded"));
    rememberGroupState(_groupDirectionField, QStringLiteral("group_direction_field_expanded"));
    rememberGroupState(_groupNeuralTracer, QStringLiteral("group_neural_tracer_expanded"));

    _groupCorrections = new QGroupBox(tr("Corrections"), this);
    auto* correctionsLayout = new QVBoxLayout(_groupCorrections);

    auto* correctionsComboRow = new QHBoxLayout();
    auto* correctionsLabel = new QLabel(tr("Active set:"), _groupCorrections);
    _comboCorrections = new QComboBox(_groupCorrections);
    _comboCorrections->setEnabled(false);
    _comboCorrections->setToolTip(tr("Choose an existing correction set to apply."));
    correctionsComboRow->addWidget(correctionsLabel);
    correctionsComboRow->addStretch(1);
    correctionsComboRow->addWidget(_comboCorrections, 1);
    correctionsLayout->addLayout(correctionsComboRow);

    _btnCorrectionsNew = new QPushButton(tr("New correction set"), _groupCorrections);
    _btnCorrectionsNew->setToolTip(tr("Create a new, empty correction set for this segmentation."));
    correctionsLayout->addWidget(_btnCorrectionsNew);

    _chkCorrectionsAnnotate = new QCheckBox(tr("Annotate corrections"), _groupCorrections);
    _chkCorrectionsAnnotate->setToolTip(tr("Toggle annotation overlay while reviewing corrections."));
    correctionsLayout->addWidget(_chkCorrectionsAnnotate);

    _groupCorrections->setLayout(correctionsLayout);
    layout->addWidget(_groupCorrections);

    _groupCustomParams = new QGroupBox(tr("Custom Params"), this);
    auto* customParamsLayout = new QVBoxLayout(_groupCustomParams);

    auto* customParamsDescription = new QLabel(
        tr("Additional JSON fields merge into the tracer params. Leave empty for defaults."), _groupCustomParams);
    customParamsDescription->setWordWrap(true);
    customParamsLayout->addWidget(customParamsDescription);

    {
        auto* profileRow = new QHBoxLayout();
        auto* profileLabel = new QLabel(tr("Profile:"), _groupCustomParams);
        _comboCustomParamsProfile = new QComboBox(_groupCustomParams);
        _comboCustomParamsProfile->addItem(tr("Custom"), QStringLiteral("custom"));
        _comboCustomParamsProfile->addItem(tr("Default"), QStringLiteral("default"));
        _comboCustomParamsProfile->addItem(tr("Robust"), QStringLiteral("robust"));
        _comboCustomParamsProfile->setToolTip(tr("Select a predefined parameter profile.\n"
                                               "- Custom: editable\n"
                                               "- Default/Robust: auto-filled and read-only"));
        profileRow->addWidget(profileLabel);
        profileRow->addWidget(_comboCustomParamsProfile, 1);
        customParamsLayout->addLayout(profileRow);
    }

    _editCustomParams = new QPlainTextEdit(_groupCustomParams);
    _editCustomParams->setToolTip(tr("Optional JSON that merges into tracer parameters before growth."));
    _editCustomParams->setPlaceholderText(QStringLiteral("{\n    \"example_param\": 1\n}"));
    _editCustomParams->setTabChangesFocus(true);
    customParamsLayout->addWidget(_editCustomParams);

    _lblCustomParamsStatus = new QLabel(_groupCustomParams);
    _lblCustomParamsStatus->setWordWrap(true);
    _lblCustomParamsStatus->setVisible(false);
    _lblCustomParamsStatus->setStyleSheet(QStringLiteral("color: #c0392b;"));
    customParamsLayout->addWidget(_lblCustomParamsStatus);

    _groupCustomParams->setLayout(customParamsLayout);
    layout->addWidget(_groupCustomParams);

    auto* buttons = new QHBoxLayout();
    _btnApply = new QPushButton(tr("Apply"), this);
    _btnApply->setToolTip(tr("Commit pending edits to the segmentation."));
    _btnReset = new QPushButton(tr("Reset"), this);
    _btnReset->setToolTip(tr("Discard pending edits and reload the segmentation state."));
    _btnStop = new QPushButton(tr("Stop tools"), this);
    _btnStop->setToolTip(tr("Exit the active editing tool and return to selection."));
    buttons->addWidget(_btnApply);
    buttons->addWidget(_btnReset);
    buttons->addWidget(_btnStop);
    layout->addLayout(buttons);

    layout->addStretch(1);

    // Signal connections extracted to SegmentationWidgetConnections.cpp for parallel compilation
    connectSignals();
}

void SegmentationWidget::syncUiState()
{
    if (_chkEditing) {
        const QSignalBlocker blocker(_chkEditing);
        _chkEditing->setChecked(_editingEnabled);
    }

    if (_lblStatus) {
        if (_editingEnabled) {
            _lblStatus->setText(_pending ? tr("Editing enabled – pending changes")
                                         : tr("Editing enabled"));
        } else {
            _lblStatus->setText(tr("Editing disabled"));
        }
    }

    if (_chkShowHoverMarker) {
        const QSignalBlocker blocker(_chkShowHoverMarker);
        _chkShowHoverMarker->setChecked(_showHoverMarker);
    }

    const bool editingActive = _editingEnabled && !_growthInProgress;

    auto updateSpin = [&](QDoubleSpinBox* spin, float value) {
        if (!spin) {
            return;
        }
        const QSignalBlocker blocker(spin);
        spin->setValue(static_cast<double>(value));
        spin->setEnabled(editingActive);
    };

    updateSpin(_spinDragRadius, _dragRadiusSteps);
    updateSpin(_spinDragSigma, _dragSigmaSteps);
    updateSpin(_spinLineRadius, _lineRadiusSteps);
    updateSpin(_spinLineSigma, _lineSigmaSteps);
    updateSpin(_spinPushPullRadius, _pushPullRadiusSteps);
    updateSpin(_spinPushPullSigma, _pushPullSigmaSteps);

    if (_groupDrag) {
        _groupDrag->setEnabled(editingActive);
    }
    if (_groupLine) {
        _groupLine->setEnabled(editingActive);
    }
    if (_groupPushPull) {
        _groupPushPull->setEnabled(editingActive);
    }

    if (_spinPushPullStep) {
        const QSignalBlocker blocker(_spinPushPullStep);
        _spinPushPullStep->setValue(static_cast<double>(_pushPullStep));
        _spinPushPullStep->setEnabled(editingActive);
    }

    if (_lblAlphaInfo) {
        _lblAlphaInfo->setEnabled(editingActive);
    }

    auto updateAlphaSpin = [&](QDoubleSpinBox* spin, float value, bool opacitySpin = false) {
        if (!spin) {
            return;
        }
        const QSignalBlocker blocker(spin);
        if (opacitySpin) {
            spin->setValue(normalizedOpacityToDisplay(value));
        } else {
            spin->setValue(static_cast<double>(value));
        }
        spin->setEnabled(editingActive);
    };

    updateAlphaSpin(_spinAlphaStart, _alphaPushPullConfig.start);
    updateAlphaSpin(_spinAlphaStop, _alphaPushPullConfig.stop);
    updateAlphaSpin(_spinAlphaStep, _alphaPushPullConfig.step);
    updateAlphaSpin(_spinAlphaLow, _alphaPushPullConfig.low, true);
    updateAlphaSpin(_spinAlphaHigh, _alphaPushPullConfig.high, true);
    updateAlphaSpin(_spinAlphaBorder, _alphaPushPullConfig.borderOffset);

    if (_spinAlphaBlurRadius) {
        const QSignalBlocker blocker(_spinAlphaBlurRadius);
        _spinAlphaBlurRadius->setValue(_alphaPushPullConfig.blurRadius);
        _spinAlphaBlurRadius->setEnabled(editingActive);
    }
    updateAlphaSpin(_spinAlphaPerVertexLimit, _alphaPushPullConfig.perVertexLimit);
    if (_chkAlphaPerVertex) {
        const QSignalBlocker blocker(_chkAlphaPerVertex);
        _chkAlphaPerVertex->setChecked(_alphaPushPullConfig.perVertex);
        _chkAlphaPerVertex->setEnabled(editingActive);
    }
    if (_alphaPushPullPanel) {
        _alphaPushPullPanel->setEnabled(editingActive);
    }

    if (_spinSmoothStrength) {
        const QSignalBlocker blocker(_spinSmoothStrength);
        _spinSmoothStrength->setValue(static_cast<double>(_smoothStrength));
        _spinSmoothStrength->setEnabled(editingActive);
    }
    if (_spinSmoothIterations) {
        const QSignalBlocker blocker(_spinSmoothIterations);
        _spinSmoothIterations->setValue(_smoothIterations);
        _spinSmoothIterations->setEnabled(editingActive);
    }

    if (_editCustomParams) {
        if (_editCustomParams->toPlainText() != _customParamsText) {
            const QSignalBlocker blocker(_editCustomParams);
            _editCustomParams->setPlainText(_customParamsText);
        }
    }

    if (_comboCustomParamsProfile) {
        const QSignalBlocker blocker(_comboCustomParamsProfile);
        const int idx = _comboCustomParamsProfile->findData(_customParamsProfile);
        if (idx >= 0) {
            _comboCustomParamsProfile->setCurrentIndex(idx);
        }
    }

    if (_editCustomParams) {
        _editCustomParams->setReadOnly(_customParamsProfile != QStringLiteral("custom"));
    }
    updateCustomParamsStatus();

    if (_spinGrowthSteps) {
        const QSignalBlocker blocker(_spinGrowthSteps);
        _spinGrowthSteps->setValue(_growthSteps);
    }

    if (_comboGrowthMethod) {
        const QSignalBlocker blocker(_comboGrowthMethod);
        int idx = _comboGrowthMethod->findData(static_cast<int>(_growthMethod));
        if (idx >= 0) {
            _comboGrowthMethod->setCurrentIndex(idx);
        }
    }

    if (_extrapolationOptionsPanel) {
        _extrapolationOptionsPanel->setVisible(_growthMethod == SegmentationGrowthMethod::Extrapolation);
    }

    if (_spinExtrapolationPoints) {
        const QSignalBlocker blocker(_spinExtrapolationPoints);
        _spinExtrapolationPoints->setValue(_extrapolationPointCount);
    }

    if (_comboExtrapolationType) {
        const QSignalBlocker blocker(_comboExtrapolationType);
        int idx = _comboExtrapolationType->findData(static_cast<int>(_extrapolationType));
        if (idx >= 0) {
            _comboExtrapolationType->setCurrentIndex(idx);
        }
    }

    // SDT params container visibility: only show when Extrapolation method AND Linear+Fit type
    if (_sdtParamsContainer) {
        _sdtParamsContainer->setVisible(
            _growthMethod == SegmentationGrowthMethod::Extrapolation &&
            _extrapolationType == ExtrapolationType::LinearFit);
    }
    if (_spinSDTMaxSteps) {
        const QSignalBlocker blocker(_spinSDTMaxSteps);
        _spinSDTMaxSteps->setValue(_sdtMaxSteps);
    }
    if (_spinSDTStepSize) {
        const QSignalBlocker blocker(_spinSDTStepSize);
        _spinSDTStepSize->setValue(static_cast<double>(_sdtStepSize));
    }
    if (_spinSDTConvergence) {
        const QSignalBlocker blocker(_spinSDTConvergence);
        _spinSDTConvergence->setValue(static_cast<double>(_sdtConvergence));
    }
    if (_spinSDTChunkSize) {
        const QSignalBlocker blocker(_spinSDTChunkSize);
        _spinSDTChunkSize->setValue(_sdtChunkSize);
    }

    // Skeleton params container visibility: only show when Extrapolation method AND SkeletonPath type
    if (_skeletonParamsContainer) {
        _skeletonParamsContainer->setVisible(
            _growthMethod == SegmentationGrowthMethod::Extrapolation &&
            _extrapolationType == ExtrapolationType::SkeletonPath);
    }
    if (_comboSkeletonConnectivity) {
        const QSignalBlocker blocker(_comboSkeletonConnectivity);
        int idx = _comboSkeletonConnectivity->findData(_skeletonConnectivity);
        if (idx >= 0) {
            _comboSkeletonConnectivity->setCurrentIndex(idx);
        }
    }
    if (_comboSkeletonSliceOrientation) {
        const QSignalBlocker blocker(_comboSkeletonSliceOrientation);
        int idx = _comboSkeletonSliceOrientation->findData(_skeletonSliceOrientation);
        if (idx >= 0) {
            _comboSkeletonSliceOrientation->setCurrentIndex(idx);
        }
    }
    if (_spinSkeletonChunkSize) {
        const QSignalBlocker blocker(_spinSkeletonChunkSize);
        _spinSkeletonChunkSize->setValue(_skeletonChunkSize);
    }
    if (_spinSkeletonSearchRadius) {
        const QSignalBlocker blocker(_spinSkeletonSearchRadius);
        _spinSkeletonSearchRadius->setValue(_skeletonSearchRadius);
    }
    // Hide fit points label and spinbox for SkeletonPath (it doesn't use polynomial fitting)
    bool showFitPoints = _extrapolationType != ExtrapolationType::SkeletonPath;
    if (_lblExtrapolationPoints) {
        _lblExtrapolationPoints->setVisible(showFitPoints);
    }
    if (_spinExtrapolationPoints) {
        _spinExtrapolationPoints->setVisible(showFitPoints);
    }

    applyGrowthDirectionMaskToUi();
    if (_chkGrowthKeybindsEnabled) {
        const QSignalBlocker blocker(_chkGrowthKeybindsEnabled);
        _chkGrowthKeybindsEnabled->setChecked(_growthKeybindsEnabled);
    }
    refreshDirectionFieldList();

    if (_directionFieldPathEdit) {
        const QSignalBlocker blocker(_directionFieldPathEdit);
        _directionFieldPathEdit->setText(_directionFieldPath);
    }
    if (_comboDirectionFieldOrientation) {
        const QSignalBlocker blocker(_comboDirectionFieldOrientation);
        int idx = _comboDirectionFieldOrientation->findData(static_cast<int>(_directionFieldOrientation));
        if (idx >= 0) {
            _comboDirectionFieldOrientation->setCurrentIndex(idx);
        }
    }
    if (_comboDirectionFieldScale) {
        const QSignalBlocker blocker(_comboDirectionFieldScale);
        int idx = _comboDirectionFieldScale->findData(_directionFieldScale);
        if (idx >= 0) {
            _comboDirectionFieldScale->setCurrentIndex(idx);
        }
    }
    if (_spinDirectionFieldWeight) {
        const QSignalBlocker blocker(_spinDirectionFieldWeight);
        _spinDirectionFieldWeight->setValue(_directionFieldWeight);
    }

    if (_comboCorrections) {
        const QSignalBlocker blocker(_comboCorrections);
        _comboCorrections->setEnabled(_correctionsEnabled && !_growthInProgress && _comboCorrections->count() > 0);
    }
    if (_chkCorrectionsAnnotate) {
        const QSignalBlocker blocker(_chkCorrectionsAnnotate);
        _chkCorrectionsAnnotate->setChecked(_correctionsAnnotateChecked);
    }
    if (_chkCorrectionsUseZRange) {
        const QSignalBlocker blocker(_chkCorrectionsUseZRange);
        _chkCorrectionsUseZRange->setChecked(_correctionsZRangeEnabled);
    }
    if (_spinCorrectionsZMin) {
        const QSignalBlocker blocker(_spinCorrectionsZMin);
        _spinCorrectionsZMin->setValue(_correctionsZMin);
    }
    if (_spinCorrectionsZMax) {
        const QSignalBlocker blocker(_spinCorrectionsZMax);
        _spinCorrectionsZMax->setValue(_correctionsZMax);
    }

    if (_lblNormalGrid) {
        const QString icon = _normalGridAvailable
            ? QStringLiteral("<span style=\"color:#2e7d32; font-size:16px;\">&#10003;</span>")
            : QStringLiteral("<span style=\"color:#c62828; font-size:16px;\">&#10007;</span>");
        const bool hasExplicitLocation = !_normalGridDisplayPath.isEmpty() && _normalGridDisplayPath != _normalGridHint;
        QString message;
        message = _normalGridAvailable ? tr("Normal grids found.") : tr("Normal grids not found.");
        if (!_normalGridHint.isEmpty()) {
            message.append(QStringLiteral(" ("));
            message.append(_normalGridHint);
            message.append(QLatin1Char(')'));
        }

        QString tooltip = message;
        if (hasExplicitLocation && !_normalGridHint.isEmpty()) {
            tooltip.append(QStringLiteral("\n"));
            tooltip.append(_normalGridHint);
        }
        if (!_volumePackagePath.isEmpty()) {
            tooltip.append(QStringLiteral("\n"));
            tooltip.append(tr("Volume package: %1").arg(_volumePackagePath));
        }

        _lblNormalGrid->setText(icon + QStringLiteral("&nbsp;") + message);
        _lblNormalGrid->setToolTip(tooltip);
        _lblNormalGrid->setAccessibleDescription(message);
    }

    if (_editNormalGridPath) {
        const bool show = _normalGridAvailable && !_normalGridPath.isEmpty();
        _editNormalGridPath->setVisible(show);
        _editNormalGridPath->setText(_normalGridPath);
        _editNormalGridPath->setToolTip(_normalGridPath);
    }

    updateNormal3dUi();

    // Approval mask checkboxes
    if (_chkShowApprovalMask) {
        const QSignalBlocker blocker(_chkShowApprovalMask);
        _chkShowApprovalMask->setChecked(_showApprovalMask);
    }
    if (_chkEditApprovedMask) {
        const QSignalBlocker blocker(_chkEditApprovedMask);
        _chkEditApprovedMask->setChecked(_editApprovedMask);
        // Edit checkboxes only enabled when show is checked
        _chkEditApprovedMask->setEnabled(_showApprovalMask);
    }
    if (_chkEditUnapprovedMask) {
        const QSignalBlocker blocker(_chkEditUnapprovedMask);
        _chkEditUnapprovedMask->setChecked(_editUnapprovedMask);
        // Edit checkboxes only enabled when show is checked
        _chkEditUnapprovedMask->setEnabled(_showApprovalMask);
    }
    if (_sliderApprovalMaskOpacity) {
        const QSignalBlocker blocker(_sliderApprovalMaskOpacity);
        _sliderApprovalMaskOpacity->setValue(_approvalMaskOpacity);
    }
    if (_lblApprovalMaskOpacity) {
        _lblApprovalMaskOpacity->setText(QString::number(_approvalMaskOpacity) + QStringLiteral("%"));
    }

    // Cell reoptimization UI state
    if (_chkCellReoptMode) {
        const QSignalBlocker blocker(_chkCellReoptMode);
        _chkCellReoptMode->setChecked(_cellReoptMode);
        // Only enabled when approval mask is visible
        _chkCellReoptMode->setEnabled(_showApprovalMask);
    }
    if (_spinCellReoptMaxSteps) {
        const QSignalBlocker blocker(_spinCellReoptMaxSteps);
        _spinCellReoptMaxSteps->setValue(_cellReoptMaxSteps);
        _spinCellReoptMaxSteps->setEnabled(_cellReoptMode);
    }
    if (_spinCellReoptMaxPoints) {
        const QSignalBlocker blocker(_spinCellReoptMaxPoints);
        _spinCellReoptMaxPoints->setValue(_cellReoptMaxPoints);
        _spinCellReoptMaxPoints->setEnabled(_cellReoptMode);
    }
    if (_spinCellReoptMinSpacing) {
        const QSignalBlocker blocker(_spinCellReoptMinSpacing);
        _spinCellReoptMinSpacing->setValue(static_cast<double>(_cellReoptMinSpacing));
        _spinCellReoptMinSpacing->setEnabled(_cellReoptMode);
    }
    if (_spinCellReoptPerimeterOffset) {
        const QSignalBlocker blocker(_spinCellReoptPerimeterOffset);
        _spinCellReoptPerimeterOffset->setValue(static_cast<double>(_cellReoptPerimeterOffset));
        _spinCellReoptPerimeterOffset->setEnabled(_cellReoptMode);
    }
    if (_comboCellReoptCollection) {
        _comboCellReoptCollection->setEnabled(_cellReoptMode);
    }
    if (_btnCellReoptRun) {
        const bool hasCollection = _comboCellReoptCollection && _comboCellReoptCollection->count() > 0;
        _btnCellReoptRun->setEnabled(_cellReoptMode && !_growthInProgress && hasCollection);
    }

    updateGrowthUiState();
}

// Note: updateNormal3dUi and setNormal3dZarrCandidates are in SegmentationWidgetNeuralTracer.cpp

QString SegmentationWidget::paramsTextForProfile(const QString& profile) const
{
    if (profile == QStringLiteral("default")) {
        // Empty => use GrowPatch defaults.
        return QString();
    }
    if (profile == QStringLiteral("robust")) {
        // See LossSettings() in core/src/GrowPatch.cpp.
        return QStringLiteral(
            "{\n"
            "  \"snap_weight\": 0.0,\n"
            "  \"normal_weight\": 0.0,\n"
            "  \"normal3dline_weight\": 1.0,\n"
            "  \"straight_weight\": 10.0,\n"
            "  \"dist_weight\": 1.0,\n"
            "  \"direction_weight\": 0.0,\n"
            "  \"sdir_weight\": 1.0,\n"
            "  \"correction_weight\": 1.0,\n"
            "  \"reference_ray_weight\": 0.0\n"
            "}\n");
    }
    return _customParamsText;
}

void SegmentationWidget::applyCustomParamsProfile(const QString& profile, bool persist, bool fromUi)
{
    const QString normalized = (profile == QStringLiteral("default") || profile == QStringLiteral("robust"))
        ? profile
        : QStringLiteral("custom");

    if (_customParamsProfile == normalized && (!fromUi || normalized == QStringLiteral("custom"))) {
        // Nothing to do.
    }
    _customParamsProfile = normalized;
    if (persist) {
        writeSetting(QStringLiteral("custom_params_profile"), _customParamsProfile);
    }

    if (_customParamsProfile != QStringLiteral("custom")) {
        _updatingCustomParamsProgrammatically = true;
        _customParamsText = paramsTextForProfile(_customParamsProfile);
        if (persist) {
            writeSetting(QStringLiteral("custom_params_text"), _customParamsText);
        }
        validateCustomParamsText();
        _updatingCustomParamsProgrammatically = false;
    }

    syncUiState();
}

// Note: restoreSettings and writeSetting are in SegmentationWidgetSettings.cpp

void SegmentationWidget::updateEditingState(bool enabled, bool notifyListeners)
{
    if (_editingEnabled == enabled) {
        return;
    }

    _editingEnabled = enabled;
    syncUiState();

    if (notifyListeners) {
        emit editingModeChanged(_editingEnabled);
    }
}

// Note: Approval mask and cell reoptimization methods (setShowHoverMarker, setShowApprovalMask,
// setEditApprovedMask, setEditUnapprovedMask, setApprovalBrushRadius, setApprovalBrushDepth,
// setApprovalMaskOpacity, setApprovalBrushColor, setCellReoptMode, setCellReoptCollections)
// are in SegmentationWidgetApprovalMask.cpp

void SegmentationWidget::setPendingChanges(bool pending)
{
    if (_pending == pending) {
        return;
    }
    _pending = pending;
    syncUiState();
}

void SegmentationWidget::setEditingEnabled(bool enabled)
{
    updateEditingState(enabled, false);
}

// Note: Tool settings methods (setDragRadius, setDragSigma, setLineRadius, setLineSigma,
// setPushPullRadius, setPushPullSigma, setPushPullStep, alphaPushPullConfig,
// setAlphaPushPullConfig, applyAlphaPushPullConfig, setSmoothingStrength,
// setSmoothingIterations) are in SegmentationWidgetToolSettings.cpp

void SegmentationWidget::handleCustomParamsEdited()
{
    if (!_editCustomParams) {
        return;
    }

    if (_updatingCustomParamsProgrammatically) {
        return;
    }

    // Edits only allowed in custom profile (UI should already be read-only otherwise).
    if (_customParamsProfile != QStringLiteral("custom")) {
        return;
    }

    _customParamsText = _editCustomParams->toPlainText();
    writeSetting(QStringLiteral("custom_params_text"), _customParamsText);
    validateCustomParamsText();
    updateCustomParamsStatus();
}

void SegmentationWidget::validateCustomParamsText()
{
    QString error;
    parseCustomParams(&error);
    _customParamsError = error;
}

void SegmentationWidget::updateCustomParamsStatus()
{
    if (!_lblCustomParamsStatus) {
        return;
    }
    if (_customParamsError.isEmpty()) {
        _lblCustomParamsStatus->clear();
        _lblCustomParamsStatus->setVisible(false);
        return;
    }
    _lblCustomParamsStatus->setText(_customParamsError);
    _lblCustomParamsStatus->setVisible(true);
}

std::optional<nlohmann::json> SegmentationWidget::parseCustomParams(QString* error) const
{
    if (error) {
        error->clear();
    }

    const QString trimmed = _customParamsText.trimmed();
    if (trimmed.isEmpty()) {
        return std::nullopt;
    }

    try {
        const QByteArray utf8 = trimmed.toUtf8();
        nlohmann::json parsed = nlohmann::json::parse(utf8.constData(), utf8.constData() + utf8.size());
        if (!parsed.is_object()) {
            if (error) {
                *error = tr("Custom params must be a JSON object.");
            }
            return std::nullopt;
        }
        return parsed;
    } catch (const nlohmann::json::parse_error& ex) {
        if (error) {
            *error = tr("Custom params JSON parse error (byte %1): %2")
                         .arg(static_cast<qulonglong>(ex.byte))
                         .arg(QString::fromStdString(ex.what()));
        }
    } catch (const std::exception& ex) {
        if (error) {
            *error = tr("Custom params JSON parse error: %1")
                         .arg(QString::fromStdString(ex.what()));
        }
    } catch (...) {
        if (error) {
            *error = tr("Custom params JSON parse error: unknown error");
        }
    }

    return std::nullopt;
}

std::optional<nlohmann::json> SegmentationWidget::customParamsJson() const
{
    QString error;
    auto parsed = parseCustomParams(&error);
    if (!error.isEmpty()) {
        return std::nullopt;
    }
    return parsed;
}

void SegmentationWidget::setGrowthMethod(SegmentationGrowthMethod method)
{
    if (_growthMethod == method) {
        return;
    }
    const int currentSteps = _growthSteps;
    if (method == SegmentationGrowthMethod::Corrections) {
        _tracerGrowthSteps = (currentSteps > 0) ? currentSteps : std::max(1, _tracerGrowthSteps);
    }
    _growthMethod = method;
    int targetSteps = currentSteps;
    if (method == SegmentationGrowthMethod::Corrections) {
        targetSteps = 0;
    } else {
        targetSteps = (currentSteps < 1) ? std::max(1, _tracerGrowthSteps) : std::max(1, currentSteps);
    }
    applyGrowthSteps(targetSteps, true, false);
    writeSetting(QStringLiteral("growth_method"), static_cast<int>(_growthMethod));
    syncUiState();
    emit growthMethodChanged(_growthMethod);
}

void SegmentationWidget::setGrowthInProgress(bool running)
{
    if (_growthInProgress == running) {
        return;
    }
    _growthInProgress = running;
    updateGrowthUiState();
}

void SegmentationWidget::setNormalGridAvailable(bool available)
{
    _normalGridAvailable = available;
    syncUiState();
}

void SegmentationWidget::setNormalGridPathHint(const QString& hint)
{
    _normalGridHint = hint;
    QString display = hint.trimmed();
    const int colonIndex = display.indexOf(QLatin1Char(':'));
    if (colonIndex >= 0 && colonIndex + 1 < display.size()) {
        display = display.mid(colonIndex + 1).trimmed();
    }
    _normalGridDisplayPath = display;
    syncUiState();
}

void SegmentationWidget::setNormalGridPath(const QString& path)
{
    _normalGridPath = path.trimmed();
    syncUiState();
}

void SegmentationWidget::setVolumePackagePath(const QString& path)
{
    _volumePackagePath = path;
    syncUiState();
}

void SegmentationWidget::setAvailableVolumes(const QVector<QPair<QString, QString>>& volumes,
                                              const QString& activeId)
{
    _volumeEntries = volumes;
    _activeVolumeId = determineDefaultVolumeId(_volumeEntries, activeId);
    if (_comboVolumes) {
        const QSignalBlocker blocker(_comboVolumes);
        _comboVolumes->clear();
        for (const auto& entry : _volumeEntries) {
            const QString& id = entry.first;
            const QString& label = entry.second.isEmpty() ? id : entry.second;
            _comboVolumes->addItem(label, id);
        }
        int idx = _comboVolumes->findData(_activeVolumeId);
        if (idx < 0 && !_volumeEntries.isEmpty()) {
            _activeVolumeId = _comboVolumes->itemData(0).toString();
            idx = 0;
        }
        if (idx >= 0) {
            _comboVolumes->setCurrentIndex(idx);
        }
        _comboVolumes->setEnabled(!_volumeEntries.isEmpty());
    }
}

void SegmentationWidget::setActiveVolume(const QString& volumeId)
{
    if (_activeVolumeId == volumeId) {
        return;
    }
    _activeVolumeId = volumeId;
    if (_comboVolumes) {
        const QSignalBlocker blocker(_comboVolumes);
        int idx = _comboVolumes->findData(_activeVolumeId);
        if (idx >= 0) {
            _comboVolumes->setCurrentIndex(idx);
        }
    }
}

void SegmentationWidget::setCorrectionsEnabled(bool enabled)
{
    if (_correctionsEnabled == enabled) {
        return;
    }
    _correctionsEnabled = enabled;
    writeSetting(QStringLiteral("corrections_enabled"), _correctionsEnabled);
    if (!enabled) {
        _correctionsAnnotateChecked = false;
        if (_chkCorrectionsAnnotate) {
            const QSignalBlocker blocker(_chkCorrectionsAnnotate);
            _chkCorrectionsAnnotate->setChecked(false);
        }
    }
    updateGrowthUiState();
}

void SegmentationWidget::setCorrectionsAnnotateChecked(bool enabled)
{
    _correctionsAnnotateChecked = enabled;
    if (_chkCorrectionsAnnotate) {
        const QSignalBlocker blocker(_chkCorrectionsAnnotate);
        _chkCorrectionsAnnotate->setChecked(enabled);
    }
    updateGrowthUiState();
}

void SegmentationWidget::setCorrectionCollections(const QVector<QPair<uint64_t, QString>>& collections,
                                                  std::optional<uint64_t> activeId)
{
    if (!_comboCorrections) {
        return;
    }
    const QSignalBlocker blocker(_comboCorrections);
    _comboCorrections->clear();
    for (const auto& pair : collections) {
        _comboCorrections->addItem(pair.second, QVariant::fromValue(static_cast<qulonglong>(pair.first)));
    }
    if (activeId) {
        int idx = _comboCorrections->findData(QVariant::fromValue(static_cast<qulonglong>(*activeId)));
        if (idx >= 0) {
            _comboCorrections->setCurrentIndex(idx);
        }
    } else {
        _comboCorrections->setCurrentIndex(-1);
    }
    _comboCorrections->setEnabled(_correctionsEnabled && !_growthInProgress && _comboCorrections->count() > 0);
}

std::optional<std::pair<int, int>> SegmentationWidget::correctionsZRange() const
{
    if (!_correctionsZRangeEnabled) {
        return std::nullopt;
    }
    return std::make_pair(_correctionsZMin, _correctionsZMax);
}

std::vector<SegmentationGrowthDirection> SegmentationWidget::allowedGrowthDirections() const
{
    std::vector<SegmentationGrowthDirection> dirs;
    if (_growthDirectionMask & kGrowDirUpBit) {
        dirs.push_back(SegmentationGrowthDirection::Up);
    }
    if (_growthDirectionMask & kGrowDirDownBit) {
        dirs.push_back(SegmentationGrowthDirection::Down);
    }
    if (_growthDirectionMask & kGrowDirLeftBit) {
        dirs.push_back(SegmentationGrowthDirection::Left);
    }
    if (_growthDirectionMask & kGrowDirRightBit) {
        dirs.push_back(SegmentationGrowthDirection::Right);
    }
    if (dirs.empty()) {
        dirs = {
            SegmentationGrowthDirection::Up,
            SegmentationGrowthDirection::Down,
            SegmentationGrowthDirection::Left,
            SegmentationGrowthDirection::Right
        };
    }
    return dirs;
}

std::vector<SegmentationDirectionFieldConfig> SegmentationWidget::directionFieldConfigs() const
{
    std::vector<SegmentationDirectionFieldConfig> configs;
    configs.reserve(_directionFields.size());
    for (const auto& config : _directionFields) {
        if (config.isValid()) {
            configs.push_back(config);
        }
    }
    return configs;
}

// Note: Direction field and growth direction methods are in SegmentationWidgetDirectionField.cpp

// Note: Neural tracer methods (setNeuralTracerEnabled, setNeuralCheckpointPath, setNeuralPythonPath,
// setNeuralVolumeScale, setNeuralBatchSize, setVolumeZarrPath) are in SegmentationWidgetNeuralTracer.cpp
